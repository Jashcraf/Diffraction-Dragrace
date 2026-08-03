"""Call-count orchestrator (Comparison F).

Counts Python-visible function calls per forward propagation for each package
(dLux measured both eager and jit) at a fixed pupil size, into
``results/callcount/``.  ``plot_results.py`` -> ``results/callcount.png``.

Usage::

    python run_callcount.py            # N = 256
    python run_callcount.py --n 512
"""

import argparse
import os
import subprocess

from run_all import ENV_PYTHON

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "bench_callcount.py")

# (package-key, extra args)
CASES = [
    ("poppy", []),
    ("hcipy", []),
    ("prysm", []),
    ("dlux", ["--dlux-mode", "eager"]),
    ("dlux", ["--dlux-mode", "jit"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Comparison F: function calls per propagation, N={args.n}")
    print("=" * 70)
    for pkg, extra in CASES:
        py = ENV_PYTHON[pkg]
        if not os.path.exists(py):
            print(f"  SKIP {pkg}: interpreter not found ({py})")
            continue
        cmd = [py, SCRIPT, "--package", pkg, "--n", str(args.n),
               "--reps", str(args.reps)] + extra
        try:
            subprocess.run(cmd, cwd=HERE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED {pkg} {extra}: exit {e.returncode}")

    print("\nCall-count runs finished. Aggregate with plot_results.py")


if __name__ == "__main__":
    main()
