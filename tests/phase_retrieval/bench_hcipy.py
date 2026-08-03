"""HCIPy phase-retrieval benchmark (no-grad only).

HCIPy is a numpy-based Fourier-optics propagator with no automatic
differentiation, so phase retrieval uses scipy L-BFGS-B with a finite-difference
gradient over the modal (Zernike) coefficients.

Run inside the hcipy env, e.g.::

    /opt/anaconda3/envs/joss/bin/python bench_hcipy.py --n 128
"""

import argparse
import warnings

import numpy as np
import hcipy

import common
import bench_util as bu


def resolve_device(device):
    """HCIPy is CPU/numpy only in this benchmark."""
    if device == "gpu":
        warnings.warn("HCIPy has no GPU backend here; using CPU.")
    return "cpu"


class HcipyForward:
    """modal coeffs -> normalised focal-plane intensity via HCIPy Fraunhofer."""

    def __init__(self, problem):
        self.p = problem
        self.pupil_grid = hcipy.make_pupil_grid(problem["n"], problem["diameter"])
        # Angular focal grid: q samples per lambda/D, matching every package.
        spatial_res = problem["wavelength"] / problem["diameter"]   # rad
        num_airy = problem["focal_pixels"] / (2 * problem["focal_q"])
        self.focal_grid = hcipy.make_focal_grid(
            problem["focal_q"], num_airy, spatial_resolution=spatial_res)
        self.prop = hcipy.FraunhoferPropagator(
            self.pupil_grid, self.focal_grid, focal_length=1.0)
        self.amp = hcipy.Field(problem["amp"].ravel(), self.pupil_grid)
        self.basis = problem["basis"]
        self.wavelength = problem["wavelength"]

    def intensity(self, coeffs):
        phase = common.phase_from_coeffs(coeffs, self.basis).ravel()
        field = self.amp * np.exp(1j * phase)
        wf = hcipy.Wavefront(hcipy.Field(field, self.pupil_grid),
                             wavelength=self.wavelength)
        img = self.prop(wf)
        return np.asarray(img.intensity.shaped, dtype=np.float64)

    def fwd(self, coeffs, target):
        I = self.intensity(coeffs)
        m = I / I.sum()
        return 0.5 * float(np.sum((m - target) ** 2))


def main():
    parser = argparse.ArgumentParser(description="HCIPy phase-retrieval benchmark")
    bu.add_common_args(parser)
    args = parser.parse_args()
    mem0 = bu.rss_peak_mb()   # post-import baseline, before building the problem

    device_used = resolve_device(args.device)
    problem = common.make_problem(args.n)
    model = HcipyForward(problem)
    # Record the focal-plane grid HCIPy actually built (may differ by +/-1 px).
    problem["focal_pixels"] = int(model.focal_grid.shape[0])

    psf_truth = model.intensity(problem["truth_coeffs"])
    assert psf_truth.dtype == np.float64, f"expected float64 PSF, got {psf_truth.dtype}"
    target = bu.normalize(psf_truth)
    x0 = np.zeros(problem["n_modes"])

    objective = lambda x: model.fwd(x, target)
    times, res, _ = bu.run_lbfgsb(objective, x0, with_grad=False,
                                  trials=args.trials)
    bu.save_result("HCIPy", "nograd", problem, device_used, times, res, res.x,
                   out_path=args.out, mem_baseline_mb=mem0,
                   extra={"device_requested": args.device,
                          "intensity_dtype": str(psf_truth.dtype)})


if __name__ == "__main__":
    main()
