"""prysm phase-retrieval benchmark.

Two modes:

* ``--grad`` (backprop): analytic reverse-mode gradient using the matrix-DFT
  adjoint ``focus_dft_adjoint`` from ``prysm.propagation`` -- the "_backprop"
  path.  This mirrors the ``ADPhaseRetireval`` class in dygdug/phase_retrieval.py.
* no ``--grad`` (nograd): forward model only, scipy L-BFGS-B estimates the
  gradient by finite differences.

Run inside the prysm env, e.g.::

    /opt/anaconda3/envs/prysm_dev/bin/python bench_prysm.py --n 128 --grad
"""

import argparse
import warnings

import numpy as np

import common
import bench_util as bu


def resolve_device(device):
    """Select the prysm math backend; fall back to CPU if GPU is unavailable."""
    if device == "gpu":
        try:
            import cupy  # noqa: F401
            from prysm.mathops import np as pnp  # noqa
            # prysm swaps its backend by reassigning the shim source module.
            import prysm.mathops as m
            m.np._srcmodule = cupy
            return "gpu"
        except Exception as e:  # pragma: no cover - no CUDA on this machine
            warnings.warn(f"prysm GPU backend unavailable ({e}); using CPU.")
    return "cpu"


class PrysmForward:
    """pupil phase (modal coeffs) -> normalised focal-plane intensity."""

    def __init__(self, problem):
        from prysm.propagation import (prepare_executor, focus_dft,
                                       focus_dft_adjoint)
        self._focus = focus_dft
        self._adjoint = focus_dft_adjoint
        self.p = problem
        self.amp = problem["amp"]
        self.basis = problem["basis"]
        # length units: metres everywhere.
        focal_dx = problem["focal_pixel_scale_rad"] * problem["efl"]
        self.executor = prepare_executor(
            pupil_dx=problem["pupil_dx"],
            pupil_samples=problem["n"],
            focal_dx=focal_dx,
            focal_samples=problem["focal_pixels"],
            wavelength=problem["wavelength"],
            efl=problem["efl"],
            kind="mdft",
        )

    def intensity(self, coeffs):
        phase = common.phase_from_coeffs(coeffs, self.basis)
        g = self.amp * np.exp(1j * phase)
        G = self._focus(g, self.executor)
        return g, G, np.abs(G) ** 2

    # --- normalised sum-of-squares cost and its dcost/dI --------------------
    def cost_and_Ibar(self, I, target):
        S = I.sum()
        m = I / S
        r = m - target                      # residual on normalised PSFs
        cost = 0.5 * np.sum(r ** 2)
        # dcost/dI_j through the normalisation S = sum_i I_i:
        Ibar = (r - np.sum(r * m)) / S
        return cost, Ibar

    def fwd(self, coeffs, target):
        _, _, I = self.intensity(coeffs)
        cost, _ = self.cost_and_Ibar(I, target)
        return cost

    def fg(self, coeffs, target):
        """Return (cost, grad) using the DFT adjoint (reverse mode)."""
        g, G, I = self.intensity(coeffs)
        cost, Ibar = self.cost_and_Ibar(I, target)
        # d(|G|^2)/dG cotangent, then propagate back to the pupil field.
        Gbar = 2 * Ibar * G
        gbar = self._adjoint(Gbar, self.executor)
        # g = amp*exp(i*phase)  =>  dcost/dphase = imag(gbar * conj(g))
        phase_bar = np.imag(gbar * np.conj(g))
        grad = np.tensordot(self.basis, phase_bar, axes=((1, 2), (0, 1)))
        return cost, grad


def main():
    parser = argparse.ArgumentParser(description="prysm phase-retrieval benchmark")
    bu.add_common_args(parser)
    parser.add_argument("--grad", action="store_true",
                        help="use analytic DFT-adjoint gradient (backprop)")
    parser.add_argument("--check-grad", action="store_true",
                        help="verify the adjoint gradient against finite diff")
    args = parser.parse_args()
    mem0 = bu.rss_peak_mb()   # post-import baseline, before building the problem

    device_used = resolve_device(args.device)
    problem = common.make_problem(args.n)
    model = PrysmForward(problem)

    # Self-consistent target (inverse crime): the same propagator generates the
    # data it will later fit, isolating propagator+optimiser speed from any
    # cross-package sampling mismatch.
    psf_truth = model.intensity(problem["truth_coeffs"])[2]
    assert psf_truth.dtype == np.float64, f"expected float64 PSF, got {psf_truth.dtype}"
    target = bu.normalize(psf_truth)

    x0 = np.zeros(problem["n_modes"])

    if args.check_grad:
        _grad_check(model, target, problem)

    if args.grad:
        objective = lambda x: model.fg(x, target)
        mode = "backprop"
    else:
        objective = lambda x: model.fwd(x, target)
        mode = "nograd"

    times, res, _ = bu.run_lbfgsb(objective, x0, with_grad=args.grad,
                                  trials=args.trials)
    bu.save_result("prysm", mode, problem, device_used, times, res, res.x,
                   out_path=args.out, mem_baseline_mb=mem0,
                   extra={"device_requested": args.device,
                          "intensity_dtype": str(psf_truth.dtype)})


def _grad_check(model, target, problem):
    x = 0.1 * np.random.default_rng(0).standard_normal(problem["n_modes"])
    cost, g_an = model.fg(x, target)
    g_fd = np.zeros_like(x)
    eps = 1e-6
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g_fd[i] = (model.fwd(xp, target) - model.fwd(xm, target)) / (2 * eps)
    rel = np.linalg.norm(g_an - g_fd) / (np.linalg.norm(g_fd) + 1e-30)
    print(f"[prysm grad-check] analytic vs finite-diff relative error = {rel:.2e}")


if __name__ == "__main__":
    main()
