"""Shared benchmark plumbing: CLI, timing, L-BFGS-B driver, result I/O.

Only depends on ``numpy`` + ``scipy`` (present in every propagator env).  The
per-package scripts supply the forward model; everything about *how* we optimise
and time is centralised here so the comparison is apples-to-apples:

    * same optimiser (scipy ``L-BFGS-B``),
    * same stopping tolerances,
    * same starting guess (zeros),
    * same normalised-PSF sum-of-squares cost,
    * same number of timed trials (median reported).
"""

import argparse
import json
import os
import platform
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

import common

# One tolerance set for every package so "time to convergence" is comparable.
LBFGSB_OPTIONS = {"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def add_common_args(parser):
    parser.add_argument("--n", type=int, default=128,
                        help="pupil pixels across (grid is n x n)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="requested compute device (gpu falls back to cpu "
                             "with a warning if unavailable)")
    parser.add_argument("--trials", type=int, default=3,
                        help="number of timed optimisation runs (median kept)")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default results/<pkg>_<mode>_n<N>.json)")
    return parser


def rss_peak_mb():
    """Peak resident set size of this process so far, in MiB.

    ``ru_maxrss`` is a monotonic high-water mark that captures *all* native
    allocations (numpy, jax, poppy, ...), unlike ``tracemalloc`` which only sees
    the Python heap.  macOS reports bytes; Linux reports kibibytes.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def normalize(psf):
    """Return an intensity PSF normalised to unit sum (numpy array)."""
    psf = np.asarray(psf, dtype=np.float64)
    s = psf.sum()
    return psf / s if s != 0 else psf


def phase_rms_error(x_opt, problem):
    """RMS phase error (radians) between the retrieved and truth phase.

    Evaluated over the illuminated pupil only.  Piston is removed because it is
    unobservable in intensity.
    """
    retrieved = common.phase_from_coeffs(x_opt, problem["basis"])
    mask = problem["amp"] > 0
    err = retrieved[mask] - problem["truth_phase"][mask]
    err = err - err.mean()
    return float(np.sqrt(np.mean(err ** 2)))


def run_lbfgsb(objective, x0, with_grad, trials):
    """Run L-BFGS-B ``trials`` times, timing each; return the median-time run.

    Parameters
    ----------
    objective : callable
        If ``with_grad`` is True, returns ``(cost, grad)``; otherwise returns a
        scalar cost and scipy estimates the gradient by finite differences.
    x0 : ndarray
        Starting coefficient vector.
    with_grad : bool
        Whether ``objective`` supplies an analytic/AD gradient.
    trials : int
        Number of timed repetitions.
    """
    times, results = [], []
    for _ in range(trials):
        t0 = perf_counter()
        res = minimize(objective, x0, jac=with_grad, method="L-BFGS-B",
                       options=LBFGSB_OPTIONS)
        times.append(perf_counter() - t0)
        results.append(res)
    order = int(np.argsort(times)[len(times) // 2])   # median index
    return times, results[order], order


def save_result(package, mode, problem, device_used, times, res, x_opt,
                out_path=None, extra=None, mem_baseline_mb=None):
    """Assemble the benchmark record and write it to JSON.

    ``mem_baseline_mb`` is the peak RSS captured at the start of the script
    (after imports, before building the problem).  When supplied, the record
    gains ``mem_peak_mb`` and ``mem_footprint_mb`` (peak minus baseline), the
    latter being the working memory the phase-retrieval problem adds on top of
    the imported libraries.
    """
    times = np.asarray(times, dtype=float)
    record = {
        "package": package,
        "mode": mode,                       # "nograd" or "backprop"
        "device_requested": None,           # filled by caller via extra if wanted
        "device_used": device_used,
        "n": problem["n"],
        "n_modes": problem["n_modes"],
        "focal_pixels": problem["focal_pixels"],
        "trials": int(times.size),
        "time_median_s": float(np.median(times)),
        "time_min_s": float(times.min()),
        "time_all_s": times.tolist(),
        "n_iter": int(getattr(res, "nit", -1)),
        "n_feval": int(getattr(res, "nfev", -1)),
        "n_jev": int(getattr(res, "njev", getattr(res, "nfev", -1))),
        "final_cost": float(np.ravel(res.fun)[0]),
        "success": bool(res.success),
        "phase_rms_error_rad": phase_rms_error(x_opt, problem),
        "truth_phase_rms_rad": float(np.sqrt(np.mean(
            problem["truth_phase"][problem["amp"] > 0] ** 2))),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    peak = rss_peak_mb()
    record["mem_peak_mb"] = peak
    if mem_baseline_mb is not None:
        record["mem_baseline_mb"] = float(mem_baseline_mb)
        record["mem_footprint_mb"] = float(peak - mem_baseline_mb)
    if extra:
        record.update(extra)

    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(
            RESULTS_DIR, f"{package}_{mode}_n{problem['n']}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    mem = (f"  mem_footprint={record['mem_footprint_mb']:.0f}MB"
           if "mem_footprint_mb" in record else "")
    print(f"[{package}/{mode}] n={problem['n']} device={device_used}  "
          f"median={record['time_median_s']*1e3:.1f} ms  "
          f"iters={record['n_iter']}  feval={record['n_feval']}  "
          f"cost={record['final_cost']:.3e}  "
          f"phaseRMSerr={record['phase_rms_error_rad']:.4f} rad{mem}  "
          f"-> {out_path}")
    return record
