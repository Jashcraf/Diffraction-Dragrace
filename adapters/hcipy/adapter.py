"""HCIPy adapter.

STATUS: exercised against hcipy 0.7.0 on macOS/arm; agrees with the internal
reference to roundoff (rel_l2 = 2.3e-15).

HCIPy's Grid/Wavefront/Propagator objects *are* its documented API, so this
adapter was already idiomatic before the contract change: build the grids and a
FraunhoferPropagator once, then call the propagator per PSF.

FFT BACKEND. HCIPy imports mkl_fft and pyfftw at import time if they are
present, and its `_math.fft` wrapper then tries `Configuration().fourier.fft.method`
in order -- default `['mkl', 'scipy', 'fftw', 'numpy']`. So "hcipy on a NumPy
config" is not what you get by leaving it alone; you get whichever accelerated
module happens to be installed in the environment. This is the adapter that
motivated the harness's backend-verification mechanism.

The method list is settable at runtime, so configure() pins it to the single
method the config asked for, and resolve_backend() reports what was pinned
rather than guessing from what is importable. An earlier version did the
guessing, which reported `pyfftw` on a `fft=numpy` config and got the run
refused -- correctly, since nothing had actually told HCIPy to use NumPy.

Worth knowing when reading these numbers: for a matrix_dft case HCIPy's
FraunhoferPropagator resolves to a MatrixFourierTransform, so no FFT runs at all
and the FFT axis is inert. `resolve_backend` records which transform class was
chosen, so a reader can tell an inert axis from an honoured one.

FREE SPACE. This adapter used to decline every plane_to_plane case on the
grounds that HCIPy "matches neither canonical kernel". That was wrong, and the
note it left behind -- identify the kernel first -- is how it got fixed.
FresnelPropagator's kernel is exp(ikz) exp(-i z/(2k) k_r^2), which is the
canonical paraxial transfer function exactly; the 5.5e-6 that looked like a
kernel mismatch was HCIPy's zero_padding=2 default computing a linear
convolution where the case pins a circular one. Pinned to the case's
convolution it scores 5.07e-16. See build() for the two overrides and what each
is worth. AngularSpectrumPropagator still declines, because its q=2 is hardcoded
rather than a default.

CENTRING. HCIPy is the one library here whose two planes disagree:
`make_pupil_grid` is interpixel (no sample at the origin) while `make_focal_grid`
puts a sample on the axis. Declaring a single convention for both costs
rel_l2 = 5.9e-3 -- small enough to be mistaken for a tolerance problem and
"fixed" by loosening the gate. Declared per plane it is exact.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import pupil_field

#: dragrace's fft_backend names -> HCIPy's Configuration method names.
_FFT_METHOD = {
    "numpy": "numpy",
    "scipy_pocketfft": "scipy",
    "mkl": "mkl",
    "pyfftw": "fftw",
}


@register("hcipy")
class HCIPyAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("hcipy",)

    #: make_pupil_grid is interpixel; make_focal_grid has a sample on the axis.
    grid_centering = {"pupil": "interpixel", "focus": "pixel"}

    #: Non-default FresnelPropagator arguments used on plane_to_plane cases.
    #: A class constant rather than something build() sets, because the worker
    #: calls resolve_backend() *before* build() -- as an instance attribute the
    #: deviation was silently recorded as null in every result, which is exactly
    #: the sort of invisible adapter choice this field exists to prevent.
    #: See build() for why each differs from HCIPy's default.
    FREE_SPACE_PARAMS = {"num_oversampling": 1, "zero_padding": 1}

    def versions(self) -> dict[str, str]:
        import hcipy
        return {"hcipy": getattr(hcipy, "__version__", "unknown"), "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("matrix_dft", "fft", "fresnel_tf", "angular_spectrum"):
            return Unsupported(f"no HCIPy path for {case.algorithm_class}")
        if case.algorithm_class == "angular_spectrum":
            # Identified, rather than guessed at. HCIPy's exact kernel is right:
            # AngularSpectrumPropagator applies exp(i k_z z) with
            # k_z = sqrt(k^2 - k_r^2), which is the canonical sqrt form. What it
            # cannot do is apply it as a *circular* convolution -- it hardcodes
            # FourierFilter(..., q=2) (propagation/angular_spectrum.py) with no
            # parameter to reach it, so it always computes the zero-padded linear
            # convolution. That is a different quantity from the one an
            # angular_spectrum case pins, and it bottoms out at 5.52e-6 against
            # the reference no matter how num_oversampling is set. Its paraxial
            # sibling FresnelPropagator *does* expose zero_padding, which is why
            # fresnel_tf runs here and angular_spectrum does not.
            return Unsupported(
                "HCIPy's AngularSpectrumPropagator hardcodes FourierFilter(q=2), so "
                "it always computes a zero-padded linear convolution where this case "
                "pins the circular one; measured floor 5.52e-6 at any num_oversampling, "
                "against a 1e-10 gate. The kernel itself is correct -- use a fresnel_tf "
                "case (fresnel_d50_z1m / fresnel_scan_z1m), where HCIPy's "
                "FresnelPropagator exposes zero_padding and scores 5.1e-16.")
        if config.is_gpu:
            return Unsupported("HCIPy has no mature GPU backend")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import hcipy

        self._want_fft = config.fft_backend
        method = _FFT_METHOD.get(config.fft_backend)
        if method is None:
            # 'native'/'xla' mean "whatever the device path uses"; leave HCIPy's
            # own preference order in place and report it.
            self._method = list(hcipy.Configuration().fourier.fft.method)
            return True

        if method == "fftw":
            try:
                import pyfftw  # noqa: F401
            except ImportError:
                return Unsupported("pyfftw not installed")
        if method == "mkl":
            try:
                import mkl_fft  # noqa: F401
            except ImportError:
                return Unsupported("mkl_fft not installed")

        # A single-element list on purpose. Leaving the fallbacks in place means
        # a method that raises is silently replaced by another one mid-run, and
        # the result would carry the label of a backend it did not use.
        hcipy.Configuration().fourier.fft.method = [method]
        self._method = [method]
        return True

    def resolve_backend(self) -> dict:
        import hcipy

        from dragrace.backend import detect_blas

        cfg = hcipy.Configuration().fourier
        active = list(cfg.fft.method)
        # Report the harness's name for whatever HCIPy will actually reach for
        # first, so `requested` and `resolved` are in the same vocabulary.
        inverse = {v: k for k, v in _FFT_METHOD.items()}
        resolved = inverse.get(active[0], active[0]) if active else "unknown"

        return {
            "array_module": "numpy",
            "fft_backend": resolved,
            "device": "cpu",
            "blas": detect_blas(),
            "hcipy_fft_method": active,
            # Non-default propagator arguments travel with the result. A reader
            # comparing this row against HCIPy's out-of-the-box behaviour has to
            # be able to see that zero_padding was pinned to the case's
            # convolution, and what that was worth.
            "hcipy_free_space_params": dict(self.FREE_SPACE_PARAMS),
            # Recorded because they change what is measured: emulate_fftshifts
            # trades ~10x accuracy for ~3x speed on the FFT path, and
            # precompute_matrices decides whether the MFT kernels are hoisted.
            "hcipy_emulate_fftshifts": bool(cfg.fft.emulate_fftshifts),
            "hcipy_mft_precompute_matrices": bool(cfg.mft.precompute_matrices),
            # Which transform class HCIPy picks depends on the grids, which do
            # not exist until build() -- and resolve_backend() runs before it.
            # `--mode ledger` shows it directly: an MFT case records no FFT
            # calls at all, which is what an inert FFT axis looks like.
        }

    def build(self, case: Case, config: Config):
        import hcipy

        if case.kind == "plane_to_plane":
            # FresnelPropagator, not AngularSpectrumPropagator: the case's
            # algorithm_class is fresnel_tf, and HCIPy's Fresnel transfer
            # function is exp(ikz) exp(-i z/(2k) (kx^2+ky^2)) which with
            # k = 2 pi / lambda is identically the canonical exp(-i pi lambda z
            # f^2). Verified 5.07e-16 against reference.free_space_transfer_function.
            #
            # Both defaults are overridden, and neither is a speed choice:
            #
            #   zero_padding=1 (default 2). HCIPy pads to compute a *linear*
            #     convolution; this case pins the circular one on the grid it
            #     declares, with the guard band already in array_samples. At the
            #     default HCIPy solves a different problem and lands at 5.5e-6
            #     against the reference -- not a worse answer, a different one.
            #     This is the same override the prysm adapter makes with Q=1 a
            #     few files over, for the same reason and against the same case.
            #   num_oversampling=1 (default 2). HCIPy averages the transfer
            #     function over 2x2 sub-positions per frequency cell; every other
            #     adapter here evaluates it pointwise. Worth 1.1e-6 on its own.
            #
            # Both are recorded in resolve_backend() so the deviation travels
            # with the result. Cost of HCIPy's defaults, measured at N=512:
            # 7.8 ms against 1.6 ms, i.e. the padded convolution is ~5x.
            extent = case.dx_pupil_m * case.n_pupil
            grid = hcipy.make_pupil_grid(case.n_pupil, extent)
            prop = hcipy.FresnelPropagator(
                grid, case.propagation.distance_m, **self.FREE_SPACE_PARAMS)
            field = hcipy.Field(
                pupil_field(case, centering=self.grid_centering).ravel(), grid)
            return {"case": case, "prop": prop,
                    "wf": hcipy.Wavefront(field, case.wavelength_m),
                    "shape": (case.n_pupil, case.n_pupil)}

        d = case.pupil.diameter_m
        pupil_grid = hcipy.make_pupil_grid(case.n_pupil, d * case.n_pupil / case.n_across)
        focal_grid = hcipy.make_focal_grid(
            q=case.q, num_airy=case.output.extent_lambda_f_d / 2.0,
            pupil_diameter=d, focal_length=case.output.focal_length_m,
            reference_wavelength=case.wavelength_m,
        )
        prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid,
                                          case.output.focal_length_m)
        field = hcipy.Field(
            pupil_field(case, centering=self.grid_centering).ravel(), pupil_grid)
        wf = hcipy.Wavefront(field, case.wavelength_m)
        return {"case": case, "prop": prop, "wf": wf, "shape": focal_grid.shape}

    def propagate(self, state):
        return state["prop"](state["wf"])

    def to_host(self, result) -> np.ndarray:
        ef = np.asarray(result.electric_field)
        n = int(np.sqrt(ef.size))
        return ef.reshape(n, n)
