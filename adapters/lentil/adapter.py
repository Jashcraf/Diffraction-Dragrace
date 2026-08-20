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
        if case.is_aperture:
            if config.is_gpu:
                return Unsupported("lentil has no GPU backend")
            from dragrace.apertures import APERTURE_MEMORY_BUDGET_BYTES
            need = self._hex_segments_peak_bytes(case)
            if need > APERTURE_MEMORY_BUDGET_BYTES:
                return Unsupported(
                    f"lentil.hex_segments materialises every segment before "
                    f"flattening: it appends {case.segmented.n_segments} full-size "
                    f"arrays to a list and then calls np.asarray on it "
                    f"(segmented.py:75), so the stack exists in full before "
                    f"np.sum reduces it. At N={case.n_pupil} that is "
                    f"{need / 2**30:.1f} GiB -- against a "
                    f"{APERTURE_MEMORY_BUDGET_BYTES / 2**30:.0f} GiB budget -- the "
                    f"figure already counts the Python list and the stacked copy "
                    f"coexisting. Measured: the N=2048 point was killed "
                    f"by the OOM reaper before it could write a result. "
                    f"flatten=True does not avoid this -- it is applied after the "
                    f"stack is built, not instead of it."
                )
            return True
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

    @staticmethod
    def _hex_segments_peak_bytes(case: Case) -> float:
        """Bytes in the (n_segments, size, size) float64 stack hex_segments builds.

        `size` is lentil's own, derived exactly as segmented.py:60 does -- it
        sizes the output from the ring count and the segment radius rather than
        from any grid the caller asks for, so this is ~12% wider than the case's
        N and the stack is correspondingly larger.
        """
        import math

        seg = case.segmented
        scale = case.n_pupil / case.pupil.diameter_m
        seg_radius_px = (seg.segment_vertex_to_vertex_m / 2.0) * scale
        seg_gap_px = seg.segment_gap_m * scale
        inner_radius = seg_radius_px * math.sqrt(3.0) / 2.0
        size = math.ceil((seg.rings * 2 + 1) * inner_radius * 2
                         + (seg.rings * 2) * seg_gap_px + 2 * 2)
        # Doubled: hex_segments holds the Python list of per-segment arrays and
        # the stacked copy np.asarray makes from it at the same moment, so the
        # peak is twice the stack rather than the stack.
        return 2.0 * float(seg.n_segments) * size * size * 8.0

    def _build_aperture(self, case: Case, config: Config):
        """lentil.hex_segments + lentil.spider, its documented aperture tools.

        Two things about lentil shape this adapter, and both are properties of
        the API rather than choices made here:

        1. Everything is in PIXELS. hex_segments takes an outscribing radius and
           a gap in pixels, not metres, so the case's geometry is converted at
           the grid scale. That is lentil's convention throughout shape.py.
        2. hex_segments SIZES ITS OWN OUTPUT. `size` is computed from the ring
           count and the segment radius (segmented.py:60), so the array comes
           back ~12% larger than the case's grid and has to be cropped to it.
           The crop is charged to lentil and timed, because the case pins the
           output grid and lentil offers no way to ask for one -- the same
           treatment POPPY's per-call wavefront rebuild gets on the propagation
           boards.

        Segments are selected by matching lentil's own centres against the
        canonical layout, as everywhere else on this board.
        """
        import numpy as np
        import lentil
        from lentil.segmented import hex_ring, hex_to_rc
        from dragrace.apertures import elt_segment_centres, select_for_centres

        seg = case.segmented
        scale = case.n_pupil / case.pupil.diameter_m          # px per metre
        seg_radius_px = (seg.segment_vertex_to_vertex_m / 2.0) * scale
        seg_gap_px = seg.segment_gap_m * scale

        # lentil enumerates: centre first, then ring by ring in hex_ring order.
        # hex_to_rc returns (row, col); rows run down the array, so the y axis is
        # negated to reach the (x, y) frame the canonical layout is written in.
        # The ELT layout is symmetric under that flip, so it cannot hide a
        # mismatch -- it only keeps the two frames comparable.
        step = seg_radius_px + seg_gap_px / 2.0
        centres_px = [(0.0, 0.0)]
        for ring in range(1, seg.rings + 1):
            for h in hex_ring(ring):
                r, c = hex_to_rc(h, step, False)
                centres_px.append((c, -r))
        centres_m = np.asarray(centres_px, dtype=float) / scale

        canonical = elt_segment_centres(case.segmented_spec())
        keep = select_for_centres(centres_m, canonical, tol=seg.segment_spacing_m * 0.25)
        drop = tuple(sorted(set(range(len(centres_m))) - set(keep.tolist())))
        if len(keep) != seg.n_segments:
            raise ValueError(
                f"lentil segment selection matched {len(keep)} of {seg.n_segments} "
                f"canonical ELT segments -- its hex_ring order or hex_to_rc "
                f"convention has moved. Failing rather than drawing the wrong "
                f"telescope."
            )

        return {"case": case, "aperture": True, "lentil": lentil,
                "rings": seg.rings, "seg_radius_px": seg_radius_px,
                "seg_gap_px": seg_gap_px, "drop": drop, "n": case.n_pupil,
                "spider_count": seg.spider_count,
                "spider_width_px": seg.spider_width_m * scale,
                "spider_offset": seg.spider_angle_offset_deg}

    def build(self, case: Case, config: Config):
        """Untimed: the Pupil plane, which is the reusable optical model."""
        import lentil

        if case.is_aperture:
            return self._build_aperture(case, config)

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
        if state.get("aperture"):
            import numpy as np

            amp = lentil.hex_segments(
                rings=state["rings"], seg_radius=state["seg_radius_px"],
                seg_gap=state["seg_gap_px"], drop=state["drop"], flatten=True)
            if state["spider_count"]:
                shape = amp.shape
                for i in range(state["spider_count"]):
                    amp = amp * lentil.spider(
                        shape, state["spider_width_px"],
                        angle=state["spider_offset"] + 60.0 * i)
            # Crop lentil's self-sized array to the case grid, centred.
            n = state["n"]
            off0 = (amp.shape[0] - n) // 2
            off1 = (amp.shape[1] - n) // 2
            return np.asarray(amp)[off0:off0 + n, off1:off1 + n]

        w = lentil.Wavefront(state["wavelength"]) * state["pupil"]
        return lentil.propagate_dft(w, pixelscale=state["pixelscale"],
                                    shape=state["shape"], oversample=1)

    def to_host(self, result) -> np.ndarray:
        """`.field` assembles lentil's internal Field list into one array.

        An aperture case has already returned a plain mask.
        """
        if isinstance(result, np.ndarray):
            return result
        return np.asarray(result.field)
