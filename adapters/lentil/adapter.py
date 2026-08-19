"""lentil adapter, through the Plane/Wavefront API lentil documents.

STATUS: exercised against lentil 0.8.9 on macOS/arm; agrees with the internal
reference exactly (rel_l2 = 0).

lentil is DFT-based, so it is zgemm-bound rather than FFT-bound. Practical
consequence for the config matrix: swapping mkl_fft in does essentially nothing
for this adapter, while swapping OpenBLAS -> MKL does. If lentil appears
insensitive to the FFT axis in the report, that is correct and expected, not a
broken adapter.

WHAT IS TIMED. lentil's documented flow is

    pupil = lentil.Pupil(amplitude=..., opd=..., pixelscale=..., focal_length=...)
    w = lentil.Wavefront(wavelength)
    w = w * pupil
    w = lentil.propagate_dft(w, pixelscale=..., shape=..., oversample=...)

The Pupil is the optical model and is reusable, so it is built once in build()
and untimed. The Wavefront is *consumed* by the propagation -- a user computing
a second PSF constructs a second Wavefront -- so `Wavefront(lambda) * pupil` is
per-PSF work and is inside the clock. That costs real time: 19.3 ms for
propagate_dft alone at N_p=1024 against 44.5 ms for the documented sequence,
because the multiply applies amplitude and OPD to a fresh array every call.

This is not lentil being charged for something its peers avoid. POPPY's
calc_psf calls input_wavefront() and re-applies every optic on each invocation
too (poppy_core.propagate_mono). Per-call model application is a design choice
these libraries make differently, and under an idiomatic-API comparison it is
part of what a user pays -- see docs/methodology.md.

An earlier version called lentil.fourier.dft2 directly. That measured the
transform rather than the library, which is not what a lentil user runs.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import circular_aperture, opd_waves


@register("lentil")
class LentilAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("lentil",)

    def versions(self) -> dict[str, str]:
        import lentil
        return {"lentil": getattr(lentil, "__version__", "unknown"), "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.kind == "plane_to_plane":
            return Unsupported(
                "lentil has no free-space propagator. Its only two propagation "
                "entry points, propagate_dft and propagate_fft, are both documented "
                "'in the far-field', and Wavefront exposes no near-field method -- "
                "lentil 0.8.9 is built for pupil-to-image models, not plane-to-plane. "
                "This is a capability gap, not an adapter gap.")
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
        """Untimed: the Pupil plane, which is the reusable optical model."""
        import lentil

        # Amplitude and OPD come from the harness rather than lentil's own
        # circle(): the case pins a hard-edged mask, and letting each library
        # rasterise its own aperture would put a rasterisation difference inside
        # a propagation comparison (docs/conventions.md). Physical units
        # throughout -- lentil works in metres, so no unit gymnastics here.
        pupil = lentil.Pupil(
            amplitude=circular_aperture(case, self.grid_centering),
            opd=opd_waves(case, self.grid_centering) * case.wavelength_m,
            pixelscale=case.dx_pupil_m,
            focal_length=case.output.focal_length_m,
            diameter=case.pupil.diameter_m,
        )
        return {"case": case, "lentil": lentil, "pupil": pupil,
                "wavelength": case.wavelength_m,
                # Physical detector sampling: (lambda*F/D)/q in metres.
                "pixelscale": case.dx_focus_m,
                "shape": int(case.n_focus)}

    def propagate(self, state):
        """One PSF, as lentil's documentation writes it.

        oversample=1 rather than lentil's default of 2: the case pins the output
        grid, and oversampling would compute a finer grid and rebin -- a
        different, more expensive calculation than the one asked for.
        """
        lentil = state["lentil"]
        w = lentil.Wavefront(state["wavelength"]) * state["pupil"]
        return lentil.propagate_dft(w, pixelscale=state["pixelscale"],
                                    shape=state["shape"], oversample=1)

    def to_host(self, result) -> np.ndarray:
        """`.field` assembles lentil's internal Field list into one array."""
        return np.asarray(result.field)
