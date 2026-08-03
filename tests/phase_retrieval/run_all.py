"""Orchestrate the phase-retrieval benchmark across conda environments.

Each propagator lives in its own environment (incompatible dependency stacks),
so we shell out to the right interpreter for each one, run the benchmark for a
sweep of pupil sizes, and let every script drop a JSON record into ``results/``.
``plot_results.py`` then aggregates them.

Two comparisons are produced, exactly as requested:

    Comparison A (no gradient back-prop, finite-difference L-BFGS-B):
        POPPY, HCIPy, prysm, dLux
    Comparison B (gradient back-prop, L-BFGS-B):
        prysm (DFT adjoint), dLux (jax value_and_grad)

Usage::

    python run_all.py                    # full sweep, CPU
    python run_all.py --ns 64 128        # custom sizes
    python run_all.py --device gpu       # GPU where available (falls back to CPU)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Map each package to the interpreter of the env where it is installed.
ENV_PYTHON = {
    "poppy": "/opt/anaconda3/envs/grater_jax/bin/python",
    "hcipy": "/opt/anaconda3/envs/joss/bin/python",
    "prysm": "/opt/anaconda3/envs/prysm_dev/bin/python",
    "dlux":  "/opt/anaconda3/envs/dlux/bin/python",
    "proper": "/opt/anaconda3/envs/corgi/bin/python",   # PROPER (forward-only)
}
SCRIPT = {
    "poppy": "bench_poppy.py",
    "hcipy": "bench_hcipy.py",
    "prysm": "bench_prysm.py",
    "dlux":  "bench_dlux.py",
}

# (package, needs --grad?, extra args) for each comparison.  dLux appears twice
# -- once jit (default) and once eager (--no-jit) -- so every figure can show
# JIT vs non-JIT dLux against the other propagators.
COMPARISON_A = [("poppy", False, []), ("hcipy", False, []),
                ("prysm", False, []),
                ("dlux", False, []), ("dlux", False, ["--no-jit"])]
COMPARISON_B = [("prysm", True, []),
                ("dlux", True, []), ("dlux", True, ["--no-jit"])]


def run_one(pkg, grad, n, device, trials, extra_args=()):
    py = ENV_PYTHON[pkg]
    if not os.path.exists(py):
        print(f"  SKIP {pkg}: interpreter not found ({py})")
        return
    cmd = [py, os.path.join(HERE, SCRIPT[pkg]),
           "--n", str(n), "--device", device, "--trials", str(trials)]
    if grad:
        cmd.append("--grad")
    cmd.extend(extra_args)
    tag = ("backprop" if grad else "nograd") + (" ".join(("",) + tuple(extra_args)))
    print(f"  -> {pkg} {tag} n={n}")
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED {pkg} (n={n}, grad={grad}): exit {e.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048, 4096],
                        help="pupil sizes to sweep (2^6 .. 2^10 by default)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--poppy-trials", type=int, default=1,
                        help="POPPY is slow; use fewer timed trials for it")
    parser.add_argument("--b-trials", type=int, default=5,
                        help="timed trials for Comparison B; its plot draws "
                             "1-sigma error bars, which need >=5 to mean much")
    args = parser.parse_args()

    print("=" * 70)
    print("Comparison A: no gradient back-prop (finite-difference L-BFGS-B)")
    print("  dLux run twice: jit and --no-jit (eager)")
    print("=" * 70)
    for n in args.ns:
        for pkg, grad, extra in COMPARISON_A:
            trials = args.poppy_trials if pkg == "poppy" else args.trials
            run_one(pkg, grad, n, args.device, trials, extra_args=extra)

    print("=" * 70)
    print("Comparison B: gradient back-prop (L-BFGS-B)")
    print("  dLux run twice: jit and --no-jit (eager); Comparison C reuses these")
    print(f"  {args.b_trials} timed trials (error bars = 1 sigma over these)")
    print("=" * 70)
    for n in args.ns:
        for pkg, grad, extra in COMPARISON_B:
            run_one(pkg, grad, n, args.device, args.b_trials, extra_args=extra)

    print("\nAll runs finished. Now aggregate with:")
    print(f"    {sys.executable} plot_results.py")


if __name__ == "__main__":
    main()
