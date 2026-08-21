"""Runs exactly one (adapter x case x config x mode) and writes result.json.

Always the innermost unit of execution, and always in its own interpreter, so
that a config's environment variables take effect before NumPy/JAX/CuPy are
imported and so that one adapter's import cannot perturb another's measurement.

Modes are separate passes over the same case on purpose:

  timing    no tracer, no tracemalloc. The only mode whose numbers go on a plot.
  memory    one iteration under tracemalloc + RSS high-water.
  ledger    one iteration with FFT/GEMM entry points instrumented.
  trace     one iteration under VizTracer (or jax.profiler). Stamped traced=true.
  gradient  the prysm-vs-dLux board.

A case carrying a `scan:` block is the one exception to "one run, one
measurement": it expands into one concrete case per array size, all measured in
this process and written to a single result.json under a `scan` block. Every
point then shares one machine fingerprint and one verified backend, so the slope
of the resulting curve is a property of the code rather than of the machine's
mood between two invocations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

SCHEMA_VERSION = 1

#: What the timed region means. Bumped whenever that definition changes, because
#: two results measured under different contracts are not comparable and nothing
#: else in the file would say so -- the schema is identical, the adapter name is
#: identical, only the meaning moved.
#:
#:   primitive-v1   the timed call was the library's transform entry point
#:                  (poppy.matrixDFT, lentil.fourier.dft2, a hand-written jnp
#:                  kernel for dLux).
#:   idiomatic-v1   the timed call is the one the library's own documentation
#:                  puts in front of a user (OpticalSystem.calc_psf,
#:                  Wavefront.focus_dft, propagate_dft, propagate_mono), with
#:                  everything the API permits hoisting already hoisted into
#:                  build(). See docs/methodology.md.
MEASUREMENT_CONTRACT = "idiomatic-v1"


def _result_skeleton(case_id: str, config_id: str, adapter_name: str, mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "case_id": case_id,
        "config_id": config_id,
        "mode": mode,
        "adapter": {"name": adapter_name},
        "status": "pending",
    }


def _measure(ad, case, config, mode: str, out_dir: Path, adapter_name: str,
             tag: str = "") -> dict:
    """Everything downstream of the case: build, first call, gate, mode dispatch.

    Split out from run() so that a scan case can call it once per array size
    against an adapter that was configured and backend-verified exactly once.
    Returns the result blocks for one concrete case rather than mutating a
    result, because a scan needs N of them side by side in one file.
    """
    import numpy as np

    from . import metrics, validate
    from .flops import ideal_work, gradient_ideal_work, efficiency, ledger as ledger_mod
    from .reference import reference_field, airy_field, loss_and_reference_gradient

    res: dict[str, Any] = {}

    # ---- build (untimed) ---------------------------------------------------
    is_grad = (mode == "gradient")
    if is_grad:
        gsup = ad.supports_gradient()
        if not gsup:
            res.update(status="unsupported", reason=getattr(gsup, "reason", "no gradient"))
            return res

    t0 = perf_counter()
    state = ad.build_gradient(case, config) if is_grad else ad.build(case, config)
    res["setup"] = {"build_s": perf_counter() - t0}

    # First call is where JIT compilation, FFTW planning and NVRTC kernel
    # compilation land. Reported, never folded into steady-state timing.
    t0 = perf_counter()
    first = ad.gradient(state) if is_grad else ad.propagate(state)
    ad.sync(first)
    res["setup"]["first_call_s"] = perf_counter() - t0

    ideal = gradient_ideal_work(case) if is_grad else ideal_work(case)
    res["flops"] = {"ideal": ideal.to_dict()}

    try:
        # ---- gradient board ------------------------------------------------
        if is_grad:
            loss, grad = first
            theta = getattr(ad, "gradient_theta", lambda s: None)(state)
            if theta is not None:
                ref_loss, ref_grad = loss_and_reference_gradient(
                    case, np.asarray(theta), getattr(ad, "grid_centering", "pixel"))
                gc = validate.compare_gradients(np.asarray(grad), ref_grad)
                res["gradient_accuracy"] = gc.to_dict()
                res["gradient_accuracy"]["reference"] = "central_differences"
                res["gradient_accuracy"]["loss_adapter"] = float(loss)
                res["gradient_accuracy"]["loss_reference"] = float(ref_loss)
                if gc.gate == "fail":
                    res.update(status="accuracy_fail",
                               reason=(f"gradient gate failed: max_rel_err={gc.max_rel_err:.3e}, "
                                       f"cos={gc.cosine_similarity:.12f}, "
                                       f"scale_ratio={gc.scale_ratio:.6f}. A scale_ratio near "
                                       f"2 or -1 indicates a Wirtinger convention mismatch."))
                    return res
            t = metrics.time_gradient(ad, state, case.execution.warmup, case.execution.repeats)
            res["timing"] = t.to_dict()
            res["memory"] = metrics.measure_memory(ad, state).to_dict() \
                if hasattr(ad, "propagate") else {}
            res["status"] = "ok"
            return res

        # ---- phase-retrieval board ------------------------------------------
        if case.is_retrieval:
            res.update(_retrieval_blocks(ad, case, state, first))
            if res.get("status") == "accuracy_fail":
                return res
            if mode in ("timing", "all"):
                t = metrics.time_propagation(ad, state, case.execution.warmup,
                                             case.execution.repeats, traced=False)
                res["timing"] = t.to_dict()
                # Cost of one forward model, which is the number that makes two
                # rows comparable: wall time here is (evaluations x per-call
                # cost) and the two factors are genuinely independent.
                nfev = (res.get("retrieval") or {}).get("n_fev")
                if nfev:
                    res["retrieval"]["seconds_per_forward_model"] = t.warm_median / nfev
            if mode in ("memory", "all"):
                res["memory"] = metrics.measure_memory(ad, state).to_dict()
            res["status"] = "ok"
            return res

        # ---- forward board: accuracy first ---------------------------------
        # complex_field() rather than to_host(): a code whose documented entry
        # point returns an intensity PSF still has to be gated on phase, and the
        # cost of asking it for the field must not land in the timing.
        centering = getattr(ad, "grid_centering", "pixel")
        quantity = getattr(ad, "output_quantity", "field")
        if case.is_aperture:
            # A drawn pupil is a real transmission mask, so neither the complex
            # field gate nor the dtype check applies: an aperture adapter returns
            # float64 by construction and gating it as a field would compare a
            # mask against a propagated wavefront.
            compare = validate.compare_aperture
        elif quantity == "intensity":
            compare = validate.compare_intensity
        else:
            compare = validate.compare
        out = ad.complex_field(state, first)
        if not case.is_aperture:
            metrics.verify_dtype(out, case)
        cmp_ = compare(out, reference_field(case, centering), case)
        res["accuracy"] = cmp_.to_dict()
        res["accuracy"]["grid_centering"] = centering
        if not case.pupil.aberration.coefficients and not case.is_aperture:
            phys = compare(out, airy_field(case, centering), case)
            res["accuracy"]["physics_check_analytic_airy"] = {
                "rel_l2": phys.rel_l2, "peak_ratio": phys.peak_ratio,
                "note": ("ungated: the analytic Airy pattern is the continuous-aperture "
                         "answer while every adapter propagates a pixelated hard-edged "
                         "mask, so ~1e-3 disagreement is expected and correct"),
            }
        if cmp_.gate == "fail":
            res.update(status="accuracy_fail", reason=validate.gate_message(cmp_, case))
            return res

        # ---- mode dispatch --------------------------------------------------
        if mode in ("timing", "all"):
            t = metrics.time_propagation(ad, state, case.execution.warmup,
                                         case.execution.repeats, traced=False)
            res["timing"] = t.to_dict()
            res["flops"]["efficiency"] = efficiency(
                ideal.flops, t.warm_median,
                flops_actual=None, roofline=None).to_dict()

        if mode in ("memory", "all"):
            res["memory"] = metrics.measure_memory(ad, state).to_dict()

        if mode in ("ledger", "all"):
            # configure() and build() are re-run INSIDE the patched context.
            # Adapters routinely cache a callable (self._fft2 = np.fft.fft2) at
            # configure time; such a reference is bound to the original function
            # and would bypass instrumentation entirely. Their calls are then
            # discarded with reset() so only the propagation is priced.
            with ledger_mod.record() as led:
                ad.configure(config)
                lstate = ad.build(case, config)
                ad.sync(ad.propagate(lstate))
                led.reset()
                ad.sync(ad.propagate(lstate))
            ad.teardown(lstate)
            res["flops"]["ledger"] = led.to_dict()
            if led.flops > 0:
                res["flops"]["algorithmic_overhead"] = led.flops / ideal.flops

        if mode == "trace":
            from . import tracing, trace_summary
            tpath = out_dir / f"trace{tag}.json"
            with tracing.trace(tpath):
                ad.sync(ad.propagate(state))
            res["trace"] = {"path": str(tpath), "tool": "viztracer", "mode": "full"}
            try:
                res["trace"].update(trace_summary.summarise(
                    tpath, Path("adapters") / adapter_name))
            except Exception as exc:                   # noqa: BLE001
                res["trace"]["summary_error"] = f"{type(exc).__name__}: {exc}"

        res["status"] = "ok"
        return res

    finally:
        try:
            ad.teardown(state)
        except Exception:                              # noqa: BLE001
            pass


def _retrieval_blocks(ad, case, state, first) -> dict:
    """Accuracy and diagnostics for one retrieval. Nothing here is timed.

    Two independent gates, because this board can fail in two unrelated ways and
    one number cannot tell them apart:

      forward_accuracy   is this code modelling the telescope the case
                         describes? Each adapter fits its OWN observed PSF,
                         generated by its own forward model at the truth
                         coefficients, so a code with a subtly wrong pupil would
                         converge perfectly onto its own private physics and the
                         coefficient gate would pass. This is the only check
                         that would notice.
      accuracy           did the optimiser find the truth? Coefficient error
                         over the observable modes.
    """
    import numpy as np

    from . import validate
    from .reference import reference_field
    from .retrieval import compare_forward_model

    out: dict[str, Any] = {}
    centering = getattr(ad, "grid_centering", "pixel")
    truth = np.asarray(reference_field(case, centering), dtype=float)

    # ---- does this code model the right optical system? --------------------
    try:
        psf = np.asarray(ad.retrieval_psf(state, truth), dtype=float)
        fwd, sign = compare_forward_model(psf, case, truth, centering)
        out["forward_accuracy"] = fwd.to_dict()
        out["forward_accuracy"]["grid_centering"] = centering
        # Which way round this code carries OPD into phase. Not a correction and
        # not a fault -- both conventions are in use -- but it decides the sign
        # of any wavefront read out of this code, so it travels with the result.
        out["forward_accuracy"]["opd_sign_convention"] = sign
        out["forward_accuracy"]["opd_sign_note"] = (
            "exp(+2i.pi.OPD/lambda), matching the harness" if sign == "+" else
            "exp(-2i.pi.OPD/lambda), opposite to the harness. The retrieved "
            "coefficients are unaffected -- this code fits an observed PSF its own "
            "forward model produced, so both conventions recover the same theta -- "
            "but a wavefront read out of this code has the opposite sign from one "
            "read out of a '+' code.")
        if fwd.gate == "fail":
            out.update(
                status="accuracy_fail",
                reason=(f"forward model FAIL: this code's PSF at the truth "
                        f"coefficients differs from the reference by "
                        f"rel_l2={fwd.rel_l2:.3e} (> "
                        f"{case.retrieval.max_forward_rel_l2:.1e}) after fitting a "
                        f"scale of {fwd.scale_abs:.6g}. It is modelling a different "
                        f"optical system, so however fast its retrieval converged, "
                        f"the time is not comparable. A peak offset of "
                        f"{fwd.peak_offset_px} px would indicate a grid-centring "
                        f"mismatch rather than wrong optics."))
            return out
    except NotImplementedError:
        out["forward_accuracy"] = {
            "note": "adapter does not expose retrieval_psf(); the forward model is "
                    "therefore ungated and this row proves only that the optimiser "
                    "converged, not that it converged on the right physics"}

    # ---- did the optimiser find the truth? ----------------------------------
    theta = np.asarray(ad.to_host(first), dtype=float).ravel()
    cmp_ = validate.compare_retrieval(theta, truth, case)
    out["accuracy"] = cmp_.to_dict()
    out["accuracy"]["grid_centering"] = centering
    out["retrieval"] = dict(ad.retrieval_report(state, first) or {})
    if cmp_.gate == "fail":
        out.update(status="accuracy_fail", reason=validate.gate_message(cmp_, case))
    return out


def _scan(ad, case, config, mode: str, out_dir: Path, adapter_name: str) -> dict:
    """Measure every point of a scan case into one `scan` block.

    A point that fails is recorded and the scan continues. The alternative --
    aborting the file -- throws away every size that did measure cleanly, and
    the usual reason a large point fails (memory) is itself the finding.
    """
    points = []
    for sub in case.scan_cases():
        value = getattr(sub, case.scan.parameter)
        point: dict[str, Any] = {
            "scan_value": value,
            "case_id": sub.id,
            "n_pupil": sub.n_pupil,
            "n_across": sub.n_across,
            "n_focus": sub.n_focus,
        }
        try:
            # supports() again, per point. run() already asked once against the
            # scan case, but that carries the template size and a scan is
            # precisely where an adapter's answer can change with N: lentil
            # materialises every segment before flattening, so the ELT board
            # costs it 8.5 GB at N=1024 and 34 GB at N=2048. Without this the
            # large point takes the process down with it and the whole file is
            # lost, including the sizes that measured cleanly -- an OOM kill
            # writes no result.json at all.
            sup = ad.supports(sub, config)
            if not sup:
                point.update(status="unsupported",
                             reason=getattr(sup, "reason", "unsupported at this size"))
                points.append(point)
                continue
            point.update(_measure(ad, sub, config, mode, out_dir, adapter_name,
                                  tag=f"_{case.scan.parameter}{value}"))
        except Exception as exc:                       # noqa: BLE001
            point.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                         traceback=traceback.format_exc())
        points.append(point)

    ok = [p for p in points if p.get("status") == "ok"]
    if len(ok) == len(points):
        status, reason = "ok", None
    elif ok:
        # Deliberately not "ok": a curve with holes in it must not be read as a
        # complete measurement just because most of its points landed.
        status = "partial"
        reason = ("scan incomplete: " + ", ".join(
            f"{p['scan_value']}={p.get('status')}" for p in points
            if p.get("status") != "ok"))
    else:
        status = points[0].get("status", "failed")
        reason = points[0].get("reason")

    res: dict[str, Any] = {
        "scan": {
            "parameter": case.scan.parameter,
            "values": [p["scan_value"] for p in points],
            "points": points,
        },
        "status": status,
    }
    if reason:
        res["reason"] = reason
    return res


def run(case, config, adapter_name: str, mode: str,
        out_dir: Path, strict_backend: bool = True) -> dict:
    import contextlib

    from . import adapter as adapter_mod
    from . import fingerprint
    from .flops import ledger as ledger_mod

    res = _result_skeleton(case.id, config.id, adapter_name, mode)
    # Recorded because the figures need it: an aperture board measures drawing,
    # not propagation, and a y-axis reading "propagation time" there is simply
    # wrong. Absent from results written before the aperture board existed, so
    # readers of this field must tolerate None.
    res["case_kind"] = case.kind
    res["machine"] = fingerprint.machine()
    res["provenance"] = fingerprint.provenance()

    ad = adapter_mod.get(adapter_name)
    res["adapter"].update({"status": ad.status, "reviewed_by": ad.reviewed_by,
                           "grid_centering": getattr(ad, "grid_centering", "pixel")})

    # ---- support and configuration ----------------------------------------
    # A pure ledger pass defers the library import into the patched region, so
    # that a code binding np.fft at import time is still intercepted rather than
    # priced at a silent zero. Every other mode imports here, where a broken
    # install becomes an honest `unsupported` instead of a traceback. Mode "all"
    # deliberately keeps the eager import: its timing pass runs first and would
    # have imported the library anyway, so its ledger block carries the same
    # caveat as any re-import -- use `--mode ledger` when the ledger is the point.
    req = ad.check_requirements(deep=(mode != "ledger"))
    if not req:
        res.update(status="unsupported", reason=getattr(req, "reason", "missing requirement"))
        return res
    sup = ad.supports(case, config)
    if not sup:
        res.update(status="unsupported",
                   reason=getattr(sup, "reason", "unsupported"))
        return res

    # For a ledger pass the instrumentation has to be installed before the
    # library is imported, and configure() is where that import happens. Several
    # of these codes capture NumPy's entry points at module scope -- PROPER's
    # prop_ptp.py does `from numpy.fft import fft2, ifft2`, HCIPy's _math/fft.py
    # closes over `getattr(np.fft, name)` -- so a ledger opened any later than
    # this prices their transforms at zero. record() is reentrant, so the
    # narrower context _measure() opens around the propagation still works and
    # still resets to drop the build's calls.
    stack = contextlib.ExitStack()
    with stack:
        if mode == "ledger":
            stack.enter_context(ledger_mod.record())
        return _run_configured(ad, case, config, mode, out_dir, adapter_name,
                               res, strict_backend)


def _run_configured(ad, case, config, mode: str, out_dir: Path, adapter_name: str,
                    res: dict, strict_backend: bool) -> dict:
    """Everything from configure() onward, so run() can wrap it in a ledger."""
    from . import backend

    conf = ad.configure(config)
    if not conf:
        res.update(status="unsupported", reason=getattr(conf, "reason", "configure failed"))
        return res

    res["adapter"]["versions"] = ad.versions()
    resolved = ad.resolve_backend()
    not_selectable = tuple(getattr(ad, "config_axes_not_selectable", ()))
    res["backend"] = backend.snapshot(config, resolved, not_selectable)
    try:
        res["backend"]["warnings"] = backend.verify(
            config, resolved, strict=strict_backend, not_selectable=not_selectable)
    except backend.BackendMismatch as exc:
        res.update(status="backend_mismatch", reason=str(exc))
        return res

    # ---- measure ------------------------------------------------------------
    # A scan case is measured point by point in this same process, against the
    # adapter configured and backend-verified above. Every point therefore
    # shares one machine fingerprint and one resolved backend, which is what
    # makes the *shape* of the curve a property of the code rather than of
    # whatever else the machine was doing between two separate runs.
    if case.is_scan:
        res.update(_scan(ad, case, config, mode, out_dir, adapter_name))
    else:
        res.update(_measure(ad, case, config, mode, out_dir, adapter_name))
    return res


def _pin_cpus(config) -> None:
    """Restrict this process to `config.threads` cores, before any import.

    The environment variables are not enough and it took a wrong number to
    notice. OMP_NUM_THREADS, MKL_NUM_THREADS and friends are honoured by
    OpenBLAS, MKL and numexpr, so every NumPy-backed adapter really does run on
    one core under `threads=1` -- measured cpu/wall = 1.00 for prysm, lentil,
    HCIPy, POPPY and the baseline. XLA honours none of them, and no XLA_FLAGS
    setting reaches its thread pool on jaxlib 0.10.2 either (see
    Config.jax_env). dLux was therefore running on ~10 cores on every board in
    this repo labelled threads=1, and on the phase-retrieval board that inverted
    the result: at N_p=1024 it beat prysm 877 ms to 1353 ms unpinned, and lost
    2117 ms to 1737 ms once both were held to one core.

    Affinity is the fix because it is a property of the PROCESS. A library
    cannot opt out of it, so it needs no per-adapter knob, no cooperation from
    XLA, and it constrains anything that sizes a pool from the core count.

    GPU configs are left alone: the device does the work and pinning the host
    thread would only throttle dispatch. Non-Linux hosts have no
    sched_setaffinity and are skipped -- the cpu/wall ratio in every timing
    block is what catches the consequence, rather than this silently not
    applying.
    """
    if config.is_gpu or not hasattr(os, "sched_setaffinity"):
        return
    try:
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:max(1, int(config.threads))]))
    except OSError:
        pass          # cgroup-restricted or otherwise refused; cpu/wall records it


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dragrace-worker")
    ap.add_argument("--case", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", default="timing",
                    choices=["timing", "memory", "ledger", "trace", "gradient", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-strict-backend", action="store_true",
                    help="Permit a resolved backend that contradicts the config. "
                         "For exploration only -- never valid for a recorded result.")
    args = ap.parse_args(argv)

    # Environment must be applied before numpy/jax/cupy are imported, so the
    # config is loaded with yaml only and nothing heavy is touched until after.
    from .config import Config
    config = Config.from_yaml(args.config)
    os.environ.update(config.full_env())
    _pin_cpus(config)

    from .case import Case
    case = Case.from_yaml(args.case)
    if config.precision_override:
        case = type(case)(**{**case.__dict__, "dtype": config.precision_override})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from . import adapter as adapter_mod
    adapter_mod.discover("adapters")

    try:
        res = run(case, config, args.adapter, args.mode, out_dir,
                  strict_backend=not args.no_strict_backend)
    except Exception as exc:                           # noqa: BLE001
        res = _result_skeleton(case.id, config.id, args.adapter, args.mode)
        res.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())

    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("status", "case_id", "config_id", "mode")
                      if k in res}))
    if res.get("status") not in ("ok", "unsupported"):
        print(res.get("reason", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
