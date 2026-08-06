"""lentil adapter.

STATUS: unverified -- written from lentil's documented API, not yet run against
an install. Check before publishing.

lentil is DFT-based, so it is zgemm-bound rather than FFT-bound. Practical
consequence for the config matrix: swapping mkl_fft in does essentially nothing
for this adapter, while swapping OpenBLAS -> MKL does. If lentil appears
insensitive to the FFT axis in the report, that is correct and expected, not a
broken adapter.

Uses lentil.fourier.dft2 directly rather than the Plane/propagate machinery, so
that what is timed is the transform rather than model assembly.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import pupil_field


@register("lentil")
class LentilAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("lentil",)

    def versions(self) -> dict[str, str]:
        import lentil
        return {"lentil": getattr(lentil, "__version__", "unknown"), "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class != "matrix_dft":
            return Unsupported(f"lentil is DFT-based; no path for {case.algorithm_class}")
        if config.is_gpu:
            return Unsupported("lentil has no GPU backend")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        return {"array_module": "numpy", "device": "cpu",
                # lentil makes little use of FFT; the BLAS is the relevant axis.
                "fft_backend": None, "blas": detect_blas()}

    def build(self, case: Case, config: Config):
        import lentil

        # dft2 evaluates sum f * exp(-2i.pi.alpha.x.u) over integer sample
        # indices, so alpha is the product of the two sample spacings expressed
        # in the harness's units: dx = 1/N_D (pupil diameters) and du = 1/q
        # (lambda F/D), giving alpha = 1/(N_D * q).
        alpha = 1.0 / (case.n_across * case.q)
        return {"case": case, "dft2": lentil.fourier.dft2, "field": pupil_field(case),
                "alpha": alpha, "npix": int(case.n_focus)}

    def propagate(self, state):
        return state["dft2"](state["field"], state["alpha"], state["npix"])
