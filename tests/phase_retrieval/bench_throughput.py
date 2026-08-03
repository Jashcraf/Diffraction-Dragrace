"""Batched-throughput benchmark (Comparison E).

A trade study evaluates *many* configurations, so the metric that matters is
**cases per second**, not the cost of a single solve.  This script times how
fast each package can evaluate a batch of ``B`` forward models (pupil -> PSF),
each with a different aberration, and reports throughput = B / wall_time.

Two batching strategies:

* ``loop``  — a plain Python ``for`` loop over cases (how a trade study runs
  today with any package). Available for all four propagators.
* ``vmap``  — ``jax.vmap`` fuses the whole batch into one compiled kernel.
  Available for dLux (jax); this is the lever that turns per-case dispatch
  overhead into vectorised throughput and that scales on a GPU.

One file, lazily importing only the requested package, so it runs unchanged in
each package's conda env::

    <env>/bin/python bench_throughput.py --package dlux --mode vmap --n 256 --batch 64
"""

import argparse
import json
import os
import platform
from time import perf_counter

import numpy as np

import common
import bench_util as bu

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "throughput")


def make_batch_coeffs(problem, batch, seed=0):
    """B distinct aberration cases: truth coefficients + small perturbations."""
    rng = np.random.default_rng(seed)
    base = problem["truth_coeffs"][None, :]
    return (base + 0.1 * rng.standard_normal((batch, problem["n_modes"]))
            ).astype(np.float64)


def timed_median(fn, trials):
    ts = []
    for _ in range(trials):
        t0 = perf_counter()
        fn()
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


# ---- per-package loop evaluators (numpy packages) ------------------------
def _loop_poppy(problem, device):
    from bench_poppy import PoppyForward, resolve_device
    dev = resolve_device(device)
    model = PoppyForward(problem)
    return (lambda coeffs: [model.intensity(c) for c in coeffs]), dev


def _loop_hcipy(problem, device):
    from bench_hcipy import HcipyForward, resolve_device
    dev = resolve_device(device)
    model = HcipyForward(problem)
    return (lambda coeffs: [model.intensity(c) for c in coeffs]), dev


def _loop_prysm(problem, device):
    from bench_prysm import PrysmForward, resolve_device
    dev = resolve_device(device)
    model = PrysmForward(problem)
    return (lambda coeffs: [model.intensity(c)[2] for c in coeffs]), dev


def _loop_proper(problem, device):
    from bench_proper import ProperForward, resolve_device
    dev = resolve_device(device)
    model = ProperForward(problem)
    return (lambda coeffs: [model.intensity(c) for c in coeffs]), dev


# ---- dLux: loop and vmap, jit or eager -----------------------------------
def _dlux_evaluators(problem, device, mode, use_jit):
    import jax
    import jax.numpy as jnp
    from bench_dlux import build_forward, resolve_device
    dev = resolve_device(device)
    intensity, _, _ = build_forward(problem, use_jit=use_jit)
    maybe_jit = jax.jit if use_jit else (lambda f: f)
    if mode == "vmap":
        vf = maybe_jit(jax.vmap(intensity))

        def run(coeffs):
            return vf(jnp.asarray(coeffs)).block_until_ready()
    else:  # per-case loop, each result materialised
        f = maybe_jit(intensity)

        def run(coeffs):
            out = None
            for c in coeffs:
                out = f(jnp.asarray(c))
                out.block_until_ready()
            return out
    return run, dev


def build_runner(package, mode, problem, device, use_jit):
    if package == "dlux":
        return _dlux_evaluators(problem, device, mode, use_jit)
    if mode != "loop":
        raise SystemExit(f"{package} only supports --mode loop")
    return {"poppy": _loop_poppy, "hcipy": _loop_hcipy,
            "prysm": _loop_prysm, "proper": _loop_proper}[package](problem, device)


def main():
    parser = argparse.ArgumentParser(description="batched-throughput benchmark")
    parser.add_argument("--package", required=True,
                        choices=["poppy", "hcipy", "prysm", "dlux", "proper"])
    parser.add_argument("--mode", default="loop", choices=["loop", "vmap"])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-jit", dest="jit", action="store_false",
                        help="dLux only: run eager (no jax.jit)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem = common.make_problem(args.n)
    coeffs = make_batch_coeffs(problem, args.batch)
    run, device_used = build_runner(args.package, args.mode, problem,
                                    args.device, args.jit)

    run(coeffs)   # warm up (JIT compile / first-touch excluded from timing)
    total = timed_median(lambda: run(coeffs), args.trials)
    throughput = args.batch / total

    # jit is only meaningful for dLux; None for the numpy packages.
    jit_flag = bool(args.jit) if args.package == "dlux" else None
    record = {
        "package": {"poppy": "POPPY", "hcipy": "HCIPy", "prysm": "prysm",
                    "dlux": "dLux", "proper": "PROPER"}[args.package],
        "mode": args.mode, "n": args.n, "batch": args.batch, "jit": jit_flag,
        "device_used": device_used, "device_requested": args.device,
        "trials": args.trials, "time_median_s": total,
        "throughput_per_s": throughput,
        "mem_peak_mb": bu.rss_peak_mb(),
        "platform": platform.platform(), "machine": platform.machine(),
    }
    jsuffix = "" if jit_flag is None else ("_jit" if jit_flag else "_nojit")
    out = args.out or os.path.join(
        RESULTS_DIR,
        f"{record['package']}_{args.mode}{jsuffix}_n{args.n}_b{args.batch}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    jtag = "" if jit_flag is None else (" jit" if jit_flag else " no-jit")
    print(f"[{record['package']}/{args.mode}{jtag}] n={args.n} B={args.batch} "
          f"device={device_used}  {total*1e3:.0f} ms  "
          f"throughput={throughput:.0f} cases/s  -> {out}")


if __name__ == "__main__":
    main()
