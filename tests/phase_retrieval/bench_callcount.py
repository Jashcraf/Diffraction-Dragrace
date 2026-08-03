"""Function-call count per forward propagation (Comparison F).

Uses ``cProfile`` to count how many Python-visible function calls each package
makes to turn a set of aberration coefficients into a PSF -- i.e. how many
"layers of code" wrap the actual diffraction math for one image simulation.

cProfile records every call to a Python or C function, so the count is exact;
it does not descend into compiled BLAS/XLA kernels (those are the math itself),
which is exactly what we want -- we are measuring framework machinery, not
FLOPs.

One file, lazily importing only the requested package::

    <env>/bin/python bench_callcount.py --package poppy --n 256
    <env>/bin/python bench_callcount.py --package dlux  --n 256 --dlux-mode jit
"""

import argparse
import cProfile
import json
import os
import platform
import pstats

import numpy as np

import common

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "callcount")


def build_eval(package, problem, dlux_mode):
    """Return (callable taking coeffs, label) that produces one PSF."""
    if package == "poppy":
        from bench_poppy import PoppyForward
        m = PoppyForward(problem)
        return (lambda c: m.intensity(c)), "POPPY"
    if package == "hcipy":
        from bench_hcipy import HcipyForward
        m = HcipyForward(problem)
        return (lambda c: m.intensity(c)), "HCIPy"
    if package == "prysm":
        from bench_prysm import PrysmForward
        m = PrysmForward(problem)
        return (lambda c: m.intensity(c)[2]), "prysm"
    if package == "dlux":
        import jax
        import jax.numpy as jnp
        from bench_dlux import build_forward
        intensity, _, _ = build_forward(problem, use_jit=(dlux_mode == "jit"))
        if dlux_mode == "jit":
            f = jax.jit(intensity)
            return (lambda c: f(jnp.asarray(c)).block_until_ready()), "dLux (jit)"
        return (lambda c: intensity(jnp.asarray(c)).block_until_ready()), "dLux (eager)"
    raise SystemExit(f"unknown package {package}")


def main():
    parser = argparse.ArgumentParser(description="call-count-per-propagation benchmark")
    parser.add_argument("--package", required=True,
                        choices=["poppy", "hcipy", "prysm", "dlux"])
    parser.add_argument("--dlux-mode", default="eager", choices=["eager", "jit"])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem = common.make_problem(args.n)
    evaluate, label = build_eval(args.package, problem, args.dlux_mode)

    coeffs = problem["truth_coeffs"]
    evaluate(coeffs)   # warm up (JIT compile / lazy first-touch excluded)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.reps):
        evaluate(coeffs)
    pr.disable()
    stats = pstats.Stats(pr)

    record = {
        "package": label.split()[0],
        "label": label,
        "n": args.n,
        "calls_per_prop": stats.total_calls / args.reps,
        "primitive_calls_per_prop": stats.prim_calls / args.reps,
        "distinct_functions": len(stats.stats),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    stem = args.package + ("_" + args.dlux_mode if args.package == "dlux" else "")
    out = args.out or os.path.join(RESULTS_DIR, f"{stem}_n{args.n}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{label}] n={args.n}  {record['calls_per_prop']:.0f} calls/propagation  "
          f"({record['distinct_functions']} distinct functions)  -> {out}")


if __name__ == "__main__":
    main()
