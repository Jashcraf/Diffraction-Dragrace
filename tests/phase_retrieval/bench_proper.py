"""PROPER (John Krist) forward model for the image-simulation comparison.

PROPER is a classical FFT-based Fresnel/Fraunhofer propagator with no automatic
differentiation, so it only takes part in the forward-model-only comparison
(Comparison G), not the phase-retrieval sweeps.  The forward model rebuilds a
PROPER wavefront each call, applies the shared Subaru amplitude + phase, images
it through a lens, and returns the focal-plane intensity.

Run inside an env that has PROPER, e.g.::

    /opt/anaconda3/envs/corgi/bin/python bench_proper.py --n 256
"""

import argparse
import warnings

import numpy as np
import proper

import common
import bench_util as bu

proper.print_it = False   # silence PROPER's per-step "Propagating" chatter


def resolve_device(device):
    """PROPER is CPU/numpy only in this benchmark."""
    if device == "gpu":
        warnings.warn("PROPER has no GPU backend here; using CPU.")
    return "cpu"


class ProperForward:
    """modal coeffs -> normalised focal-plane intensity via PROPER."""

    def __init__(self, problem):
        self.p = problem
        self.amp = problem["amp"]
        self.basis = problem["basis"]
        self.gridsize = problem["n"]
        self.diameter = problem["diameter"]
        self.wavelength = problem["wavelength"]
        self.efl = problem["efl"]

    def intensity(self, coeffs):
        phase = common.phase_from_coeffs(coeffs, self.basis)
        opd = phase * self.wavelength / (2 * np.pi)          # radians -> metres
        # beam_diam_fraction = 1.0: the shared aperture fills the whole grid.
        wf = proper.prop_begin(self.diameter, self.wavelength,
                               self.gridsize, 1.0)
        proper.prop_multiply(wf, self.amp)
        proper.prop_add_phase(wf, opd)
        proper.prop_define_entrance(wf)
        proper.prop_lens(wf, self.efl)
        proper.prop_propagate(wf, self.efl)
        field, _ = proper.prop_end(wf)
        return np.abs(field) ** 2


def main():
    parser = argparse.ArgumentParser(description="PROPER forward-model benchmark")
    bu.add_common_args(parser)
    args = parser.parse_args()

    device_used = resolve_device(args.device)
    problem = common.make_problem(args.n)
    model = ProperForward(problem)
    psf = model.intensity(problem["truth_coeffs"])
    assert psf.dtype == np.float64, f"expected float64 PSF, got {psf.dtype}"
    print(f"[PROPER] n={args.n} device={device_used}  psf {psf.shape} "
          f"dtype={psf.dtype}  sum={psf.sum():.3e}")


if __name__ == "__main__":
    main()
