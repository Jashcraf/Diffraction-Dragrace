"""HCIPy adapter.

STATUS: unverified -- written from HCIPy's documented API, not yet run against
an install. Check before publishing.

HCIPy's FFT layer probes for mkl_fft and pyfftw at import and uses them if
present, without saying so. resolve_backend() below reports what it actually
found, and the worker fails the run if that contradicts the config. This is the
adapter that motivated that whole mechanism.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import pupil_field


@register("hcipy")
class HCIPyAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("hcipy",)

    def versions(self) -> dict[str, str]:
        import hcipy
        return {"hcipy": getattr(hcipy, "__version__", "unknown"), "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("matrix_dft", "fft", "fresnel_tf", "angular_spectrum"):
            return Unsupported(f"no HCIPy path for {case.algorithm_class}")
        if config.is_gpu:
            return Unsupported("HCIPy has no mature GPU backend")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        self._want_fft = config.fft_backend
        if config.fft_backend == "pyfftw":
            try:
                import pyfftw  # noqa: F401
            except ImportError:
                return Unsupported("pyfftw not installed")
        if config.fft_backend == "mkl":
            try:
                import mkl_fft  # noqa: F401
            except ImportError:
                return Unsupported("mkl_fft not installed")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        found = "numpy"
        try:
            # HCIPy selects its FFT front end at import; report whichever of the
            # accelerated modules is actually importable in this interpreter,
            # since that is what it will have picked up.
            import mkl_fft  # noqa: F401
            found = "mkl"
        except ImportError:
            try:
                import pyfftw  # noqa: F401
                found = "pyfftw"
            except ImportError:
                found = "numpy"
        return {"array_module": "numpy", "fft_backend": found,
                "device": "cpu", "blas": detect_blas(),
                "note": "HCIPy resolves its FFT module at import; verify against "
                        "hcipy.fourier internals for the installed version"}

    def build(self, case: Case, config: Config):
        import hcipy

        d = case.pupil.diameter_m
        pupil_grid = hcipy.make_pupil_grid(case.n_pupil, d * case.n_pupil / case.n_across)
        focal_grid = hcipy.make_focal_grid(
            q=case.q, num_airy=case.output.extent_lambda_f_d / 2.0,
            pupil_diameter=d, focal_length=case.output.focal_length_m,
            reference_wavelength=case.wavelength_m,
        )
        prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid,
                                          case.output.focal_length_m)
        field = hcipy.Field(pupil_field(case).ravel(), pupil_grid)
        wf = hcipy.Wavefront(field, case.wavelength_m)
        return {"case": case, "prop": prop, "wf": wf, "shape": focal_grid.shape}

    def propagate(self, state):
        return state["prop"](state["wf"])

    def to_host(self, result) -> np.ndarray:
        ef = np.asarray(result.electric_field)
        n = int(np.sqrt(ef.size))
        return ef.reshape(n, n)
