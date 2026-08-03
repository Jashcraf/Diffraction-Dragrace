"""Memory-footprint comparison of the phase-retrieval problem (Comparison D).

Runs the seven requested cases over N = 2^6 .. 2^10 and records the peak RSS
above a post-import baseline (see ``bench_util.rss_peak_mb``).  Results go to
``results/mem/`` so they do not overwrite the timing sweep, and
``plot_results.py`` turns them into ``results/memory_footprint.png``.

The seven lines:
    1. POPPY                        (finite difference)
    2. HCIPy                        (finite difference)
    3. prysm  (finite difference)
    4. prysm  (back-prop, DFT adjoint)
    5. dLux   (finite difference, no jit / eager)
    6. dLux   (back-prop, no jit / eager)
    7. dLux   (back-prop, jit)

Memory is deterministic, so a single trial per point is enough.

Usage::

    python run_mem.py                 # N = 64..1024
    python run_mem.py --ns 64 128     # custom
"""

import argparse
import os
import subprocess

from run_all import ENV_PYTHON, SCRIPT

HERE = os.path.dirname(os.path.abspath(__file__))
MEM_DIR = os.path.join(HERE, "results", "mem")

# (label, package-key, extra CLI args, output-file stem)
CASES = [
    ("POPPY",                     "poppy", [],                    "POPPY_nograd"),
    ("HCIPy",                     "hcipy", [],                    "HCIPy_nograd"),
    ("prysm (finite diff)",       "prysm", [],                    "prysm_nograd"),
    ("prysm (back-prop)",         "prysm", ["--grad"],            "prysm_backprop"),
    ("dLux (finite diff, no jit)", "dlux", ["--no-jit"],          "dLux_nograd_nojit"),
    ("dLux (back-prop, no jit)",  "dlux",  ["--grad", "--no-jit"], "dLux_backprop_nojit"),
    ("dLux (back-prop, jit)",     "dlux",  ["--grad"],            "dLux_backprop_jit"),
]


def run_case(label, pkg, extra, stem, n, device):
    py = ENV_PYTHON[pkg]
    if not os.path.exists(py):
        print(f"  SKIP {label}: interpreter not found ({py})")
        return
    out = os.path.join(MEM_DIR, f"{stem}_n{n}.json")
    cmd = [py, os.path.join(HERE, SCRIPT[pkg]), "--n", str(n),
           "--device", device, "--trials", "1", "--out", out] + extra
    print(f"  -> {label}  n={n}")
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED {label} (n={n}): exit {e.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024],
                        help="pupil sizes to sweep (2^6 .. 2^10 by default)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    os.makedirs(MEM_DIR, exist_ok=True)
    print("=" * 70)
    print("Comparison D: memory footprint (peak RSS above import baseline)")
    print("=" * 70)
    for n in args.ns:
        for label, pkg, extra, stem in CASES:
            run_case(label, pkg, extra, stem, n, args.device)

    print("\nMemory runs finished. Aggregate with plot_results.py")


if __name__ == "__main__":
    main()
