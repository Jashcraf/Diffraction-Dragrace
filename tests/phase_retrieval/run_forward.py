"""Forward-model-only speed comparison (Comparison G).

Like Comparison A, but with *no* phase retrieval — just the cost of a single
image simulation (pupil -> PSF) as a function of pupil size N.  Reuses
``bench_throughput.py`` at ``--batch 1`` (which times exactly one forward
propagation) so there is no separate timing path to maintain.

Six lines — POPPY, HCIPy, prysm, PROPER (John Krist), and dLux shown both jit
(warm, compiled) and eager (--no-jit).  Sweeps to N = 2048 (larger than the
retrieval sweeps, since a single propagation is cheap).  Results go to
``results/forward/``.

Usage::

    python run_forward.py               # N = 2^6 .. 2^11
    python run_forward.py --ns 64 128
"""

import argparse
import os
import subprocess

from run_all import ENV_PYTHON

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "bench_throughput.py")
LABEL = {"poppy": "POPPY", "hcipy": "HCIPy", "prysm": "prysm", "dlux": "dLux"}

# (package, extra args, output stem).  dLux appears twice: jit and eager.
CASES = [
    ("poppy", [], "POPPY"),
    ("hcipy", [], "HCIPy"),
    ("prysm", [], "prysm"),
    ("proper", [], "PROPER"),
    ("dlux", [], "dLux_jit"),
    ("dlux", ["--no-jit"], "dLux_nojit"),
]


def run_case(pkg, extra, stem, n, device, trials):
    py = ENV_PYTHON[pkg]
    if not os.path.exists(py):
        print(f"  SKIP {pkg}: interpreter not found ({py})")
        return
    out = os.path.join(HERE, "results", "forward", f"{stem}_n{n}.json")
    cmd = [py, SCRIPT, "--package", pkg, "--mode", "loop", "--n", str(n),
           "--batch", "1", "--device", device, "--trials", str(trials),
           "--out", out] + extra
    print(f"  -> {stem}  N={n}")
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED {stem} (N={n}): exit {e.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    print("=" * 70)
    print("Comparison G: forward-model (image simulation) speed, no retrieval")
    print("  dLux shown jit and eager (--no-jit)")
    print("=" * 70)
    for n in args.ns:
        for pkg, extra, stem in CASES:
            run_case(pkg, extra, stem, n, args.device, args.trials)

    print("\nForward-model runs finished. Aggregate with plot_results.py")


if __name__ == "__main__":
    main()
