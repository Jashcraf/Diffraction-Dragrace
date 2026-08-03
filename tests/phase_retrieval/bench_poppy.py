"""POPPY phase-retrieval benchmark (no-grad only).

POPPY is a classical Fraunhofer/Fresnel FFT+MFT propagator with no automatic
differentiation, so phase retrieval uses scipy L-BFGS-B with a finite-difference
gradient over the modal (Zernike) coefficients.

Run inside the poppy env, e.g.::

    /opt/anaconda3/envs/grater_jax/bin/python bench_poppy.py --n 128
"""

import argparse
import logging
import warnings

import numpy as np
import astropy.units as u
import poppy

import common
import bench_util as bu

logging.getLogger("poppy").setLevel(logging.ERROR)
_ARCSEC_PER_RAD = 206264.80624709636


def resolve_device(device):
    """POPPY GPU path is via cupy in poppy.accel_math; fall back to CPU."""
    if device == "gpu":
        try:
            import poppy.accel_math as am
            if getattr(am, "_CUPY_AVAILABLE", False):
                am.update_math_settings()  # honour POPPY_USE_CUPY if set
                if am._USE_CUPY:
                    return "gpu"
        except Exception:
            pass
        warnings.warn("POPPY GPU (cupy) unavailable; using CPU.")
    return "cpu"


class PoppyForward:
    """modal coeffs -> normalised focal-plane intensity via POPPY."""

    def __init__(self, problem):
        self.p = problem
        self.amp = problem["amp"]
        self.basis = problem["basis"]
        self.pupil_dx = problem["pupil_dx"]
        self.wavelength = problem["wavelength"]
        self.det_pixelscale = (problem["focal_pixel_scale_rad"]
                               * _ARCSEC_PER_RAD)          # arcsec / pixel
        self.fov_pixels = problem["focal_pixels"]

    def intensity(self, coeffs):
        phase = common.phase_from_coeffs(coeffs, self.basis)
        opd = phase * self.wavelength / (2 * np.pi)        # radians -> metres
        pupil = poppy.ArrayOpticalElement(
            transmission=self.amp, opd=opd,
            pixelscale=self.pupil_dx * u.m / u.pixel, name="Subaru pupil")
        osys = poppy.OpticalSystem(npix=self.p["n"], oversample=1,
                                   pupil_diameter=self.p["diameter"] * u.m)
        osys.add_pupil(pupil)
        osys.add_detector(pixelscale=self.det_pixelscale,
                          fov_pixels=self.fov_pixels, oversample=1)
        psf = osys.calc_psf(wavelength=self.wavelength, normalize="first")
        return np.asarray(psf[0].data, dtype=np.float64)

    def fwd(self, coeffs, target):
        I = self.intensity(coeffs)
        m = I / I.sum()
        return 0.5 * float(np.sum((m - target) ** 2))


def main():
    parser = argparse.ArgumentParser(description="POPPY phase-retrieval benchmark")
    bu.add_common_args(parser)
    args = parser.parse_args()
    mem0 = bu.rss_peak_mb()   # post-import baseline, before building the problem

    device_used = resolve_device(args.device)
    problem = common.make_problem(args.n)
    model = PoppyForward(problem)

    psf_truth = model.intensity(problem["truth_coeffs"])
    assert psf_truth.dtype == np.float64, f"expected float64 PSF, got {psf_truth.dtype}"
    target = bu.normalize(psf_truth)
    x0 = np.zeros(problem["n_modes"])

    objective = lambda x: model.fwd(x, target)
    times, res, _ = bu.run_lbfgsb(objective, x0, with_grad=False,
                                  trials=args.trials)
    bu.save_result("POPPY", "nograd", problem, device_used, times, res, res.x,
                   out_path=args.out, mem_baseline_mb=mem0,
                   extra={"device_requested": args.device,
                          "poppy_double_precision": bool(poppy.conf.double_precision),
                          "intensity_dtype": str(psf_truth.dtype)})


if __name__ == "__main__":
    main()
