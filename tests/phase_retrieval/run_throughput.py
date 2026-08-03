"""Batched-throughput orchestrator (Comparison E).

Sweeps batch size for each package at a fixed pupil size and records
cases/second into ``results/throughput/``.  ``plot_results.py`` turns them into
``results/throughput.png``.

Five lines:
    POPPY (loop), HCIPy (loop), prysm (loop), dLux (loop), dLux (vmap)

Usage::

    python run_throughput.py                      # N=256, B in 1..256, CPU
    python run_throughput.py --device gpu         # GPU where available
"""

import argparse
import os
import subprocess

from run_all import ENV_PYTHON

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "bench_throughput.py")

# (package-key, mode, extra args).  dLux loop shown jit and eager.
CASES = [("poppy", "loop", []), ("hcipy", "loop", []), ("prysm", "loop", []),
         ("dlux", "loop", []), ("dlux", "loop", ["--no-jit"]),
         ("dlux", "vmap", [])]


def run_case(pkg, mode, extra, n, batch, device, trials):
    py = ENV_PYTHON[pkg]
    if not os.path.exists(py):
        print(f"  SKIP {pkg}/{mode}: interpreter not found ({py})")
        return
    cmd = [py, SCRIPT, "--package", pkg, "--mode", mode, "--n", str(n),
           "--batch", str(batch), "--device", device, "--trials", str(trials)] + extra
    print(f"  -> {pkg}/{mode} {' '.join(extra)}  N={n} B={batch}")
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED {pkg}/{mode} (B={batch}): exit {e.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--poppy-trials", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Comparison E: batched throughput (cases/s), N={args.n}")
    print("=" * 70)
    for batch in args.batches:
        for pkg, mode, extra in CASES:
            trials = args.poppy_trials if pkg == "poppy" else args.trials
            run_case(pkg, mode, extra, args.n, batch, args.device, trials)

    print("\nThroughput runs finished. Aggregate with plot_results.py")


if __name__ == "__main__":
    main()
