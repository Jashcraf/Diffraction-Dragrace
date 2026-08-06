"""POPPY adapter.

STATUS: unverified -- written from POPPY's documented API, not yet run against
an install. Check before publishing.

Uses poppy.matrixDFT directly rather than OpticalSystem.calc_psf, because
calc_psf returns an intensity HDUList and bundles detector binning and
normalisation into the measurement. The MFT is the propagation this case is
about.

POPPY carries astropy.units through much of its optical machinery. That is a
usability and safety choice with a real runtime cost, and the trace-category
breakdown will surface it (`units: N% of self time`). Report it as an
attribution, not a verdict -- see docs/methodology.md.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import pupil_field


@register("poppy")
class PoppyAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("poppy", "astropy")

    def versions(self) -> dict[str, str]:
        import poppy
        import astropy
        return {"poppy": getattr(poppy, "__version__", "unknown"),
                "astropy": astropy.__version__, "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("matrix_dft", "fft", "fresnel_tf"):
            return Unsupported(f"no POPPY path for {case.algorithm_class}")
        if config.is_gpu:
            try:
                import cupy  # noqa: F401
            except ImportError:
                return Unsupported("GPU config requires CuPy (env dragrace-gpu-cupy)")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import poppy

        self._gpu = config.is_gpu
        self._fft_name = config.fft_backend
        # Toggle names have moved between POPPY versions; set what exists and
        # report what resolved rather than assuming either.
        for attr, value in (("use_fftw", config.fft_backend == "pyfftw"),
                            ("use_cuda", config.is_gpu),
                            ("use_numexpr", False)):
            if hasattr(poppy.conf, attr):
                try:
                    setattr(poppy.conf, attr, value)
                except Exception:                    # noqa: BLE001
                    pass
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        resolved = {"array_module": "numpy", "device": "cpu",
                    "fft_backend": self._fft_name, "blas": detect_blas()}
        try:
            from poppy import accel_math
            resolved["accel_math"] = {
                k: bool(getattr(accel_math, k, False))
                for k in ("_USE_FFTW", "_USE_CUPY", "_USE_NUMEXPR", "_USE_OPENCL")
                if hasattr(accel_math, k)
            }
            if resolved["accel_math"].get("_USE_CUPY"):
                resolved["array_module"] = "cupy"
                resolved["device"] = "cuda"
        except Exception:                            # noqa: BLE001
            pass
        return resolved

    def build(self, case: Case, config: Config):
        from poppy.matrixDFT import MatrixFourierTransform

        mft = MatrixFourierTransform()
        return {
            "case": case, "mft": mft, "field": pupil_field(case),
            # POPPY's nlamD is the output region size in lambda/D, npix its
            # sample count -- so q = npix / nlamD, matching the case exactly.
            "nlamD": float(case.output.extent_lambda_f_d),
            "npix": int(case.n_focus),
        }

    def propagate(self, state):
        return state["mft"].perform(state["field"], state["nlamD"], state["npix"])

    def sync(self, result) -> None:
        if self._gpu:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except ImportError:
                pass

    def to_host(self, result) -> np.ndarray:
        try:
            import cupy as cp
            if isinstance(result, cp.ndarray):
                return cp.asnumpy(result)
        except ImportError:
            pass
        return np.asarray(result)
