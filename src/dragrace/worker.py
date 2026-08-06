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


def _result_skeleton(case_id: str, config_id: str, adapter_name: str, mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "config_id": config_id,
        "mode": mode,
        "adapter": {"name": adapter_name},
        "status": "pending",
    }


def run(case, config, adapter_name: str, mode: str,
        out_dir: Path, strict_backend: bool = True) -> dict:
    import numpy as np

    from . import adapter as adapter_mod
    from . import backend, fingerprint, metrics, validate
    from .flops import ideal_work, gradient_ideal_work, efficiency, ledger as ledger_mod
    from .reference import reference_field, airy_field, loss_and_reference_gradient

    res = _result_skeleton(case.id, config.id, adapter_name, mode)
    res["machine"] = fingerprint.machine()
    res["provenance"] = fingerprint.provenance()

    ad = adapter_mod.get(adapter_name)
    res["adapter"].update({"status": ad.status, "reviewed_by": ad.reviewed_by})

    # ---- support and configuration ----------------------------------------
    req = ad.check_requirements()
    if not req:
        res.update(status="unsupported", reason=getattr(req, "reason", "missing requirement"))
        return res
    sup = ad.supports(case, config)
    if not sup:
        res.update(status="unsupported",
                   reason=getattr(sup, "reason", "unsupported"))
        return res
    conf = ad.configure(config)
    if not conf:
        res.update(status="unsupported", reason=getattr(conf, "reason", "configure failed"))
        return res

    res["adapter"]["versions"] = ad.versions()
    resolved = ad.resolve_backend()
    res["backend"] = backend.snapshot(config, resolved)
    try:
        res["backend"]["warnings"] = backend.verify(config, resolved, strict=strict_backend)
    except backend.BackendMismatch as exc:
        res.update(status="backend_mismatch", reason=str(exc))
        return res

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
                ref_loss, ref_grad = loss_and_reference_gradient(case, np.asarray(theta))
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

        # ---- forward board: accuracy first ---------------------------------
        out = ad.to_host(first)
        metrics.verify_dtype(out, case)
        cmp_ = validate.compare(out, reference_field(case), case)
        res["accuracy"] = cmp_.to_dict()
        if not case.pupil.aberration.coefficients:
            phys = validate.compare(out, airy_field(case), case)
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
            tpath = out_dir / "trace.json"
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
