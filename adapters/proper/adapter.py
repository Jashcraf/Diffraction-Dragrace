"""PROPER (PyPROPER3) adapter.

STATUS: unverified -- written against the local PROPER 3.3.4 source, not yet run.
Check before publishing.

PROPER's unit of work is a prescription function executed against module-level
global state (prop_begin / prop_end bracket a run). There is essentially nothing
to hoist into build() beyond FFTW wisdom, so its cold and warm numbers will
nearly coincide. That is a property of the API, not a measurement artifact, and
the report should say so rather than presenting PROPER as uniquely slow to warm
up.

Sampling is controlled by beam_ratio = (beam diameter)/(grid diameter), which
makes samples per lambda*F/D equal to 1/beam_ratio -- so beam_ratio = 1/q.

FFT backends, both confirmed present in proper/ on this machine:
  prop_use_fftw()  -> pyFFTW, with prop_fftw_wisdom / prop_load_fftw_wisdom
  prop_use_ffti()  -> Intel MKL, loaded by ctypes from $MKL_DIR/libmkl_rt.so

WAVELENGTH UNITS. prop_begin documents `lamda : Wavelength in meters`, and this
adapter used to pass microns. That is not a cosmetic slip: PROPER derives the
Rayleigh range z_R = pi w0^2 / lamda from it and switches propagation branch on
the result. A wavelength 1e6 too large made z_R 1e6 too small, which threw every
propagation into PROPER's far-field branch. On a pupil-to-focus case that
accidentally produced the right answer -- the far-field branch IS the Fourier
transform the case wants, so it scored 1.79e-8 while reporting a focal sampling
of "15.82 m" -- and on a free-space case it was catastrophic, rescaling the
output grid by 1e6 and scoring rel_l2 = 96.8. Corrected to metres below.

Consequence worth stating plainly: with the units right, the focal case scores
7.5e-5 rather than 1.79e-8, because PROPER then does what it was asked to do --
a genuine near-field Fresnel propagation over the focal length, which is not the
exact Fourier transform the case gates against. The old number was right for the
wrong reason.

The Intel path needs the UNVERSIONED soname; conda and PyPI ship only
libmkl_rt.so.2, so scripts/setup_env.sh creates the symlink. Without it
prop_use_ffti() fails to load even with MKL correctly installed.
"""
from __future__ import annotations

import os

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import circular_aperture, opd_waves


def _prescription_free_space(wavelength, gridsize, PASSVALUE=None):
    """Plane to plane: aperture, then straight propagation. PROPER's native mode.

    No lens and no focal plane, so this is the shortest prescription PROPER can
    execute -- which is the point of the free-space board.
    """
    import proper

    p = PASSVALUE or {}
    wfo = proper.prop_begin(p["diam_m"], wavelength, gridsize, p["beam_ratio"])
    proper.prop_multiply(wfo, p["mask"])
    if p.get("opd_m") is not None:
        proper.prop_add_phase(wfo, p["opd_m"])
    proper.prop_define_entrance(wfo)
    proper.prop_propagate(wfo, p["distance_m"])
    return proper.prop_end(wfo, NOABS=True)


def _prescription(wavelength, gridsize, PASSVALUE=None):
    """Pupil -> focus through a perfect lens. Module-level by necessity:
    PROPER executes prescriptions by name against global state.

    The aperture arrives as an array through prop_multiply rather than from
    prop_circular_aperture. PROPER's own aperture is ANTIALIASED -- 47 distinct
    edge values at N_D=128 against the harness mask's 2 -- and an antialiased
    edge is a modelling choice each of these codes makes differently, as well as
    one that costs more to render. Letting PROPER draw its own would put a
    rasterisation difference inside a propagation comparison, which is what
    docs/conventions.md exists to prevent. prop_multiply documents its argument
    as "centered at pixel (n/2, n/2)", which is the harness's `pixel` convention,
    and it applies the FFT shift itself.
    """
    import proper

    p = PASSVALUE or {}
    wfo = proper.prop_begin(p["diam_m"], wavelength, gridsize, p["beam_ratio"])
    proper.prop_multiply(wfo, p["mask"])
    if p.get("opd_m") is not None:
        proper.prop_add_phase(wfo, p["opd_m"])
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, p["efl_m"])
    proper.prop_propagate(wfo, p["efl_m"])
    return proper.prop_end(wfo, NOABS=True)


@register("proper")
class ProperAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("proper",)

    #: prop_end returns intensity unless NOABS is set; the field it can return
    #: carries a residual quadratic phase from the lens propagation. See
    #: docs/methodology.md.
    output_quantity = "intensity"

    def versions(self) -> dict[str, str]:
        import proper
        return {"PyPROPER3": getattr(proper, "__version__", "3.3.4"),
                "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.is_aperture:
            if config.is_gpu:
                return Unsupported("PROPER has no GPU backend")
            return True
        if case.algorithm_class == "angular_spectrum":
            return Unsupported(
                "PROPER's prop_propagate applies the paraxial Fresnel transfer "
                "function: it matches an internal_fresnel_tf reference to 1.02e-14 "
                "and an exact angular-spectrum one only to 4.07e-6. Use a "
                "fresnel_tf case (fresnel_d50_z1m).")
        if case.algorithm_class not in ("fft", "fresnel_tf"):
            return Unsupported(
                f"PROPER is FFT/Fresnel-based; no matrix-DFT path for "
                f"{case.algorithm_class}. Sampling is set by beam_ratio, so it "
                f"cannot hit an arbitrary focal grid the way an MFT can."
            )
        if config.is_gpu:
            return Unsupported("PROPER has no GPU backend")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import proper

        self._fft_name = config.fft_backend
        if config.fft_backend == "pyfftw":
            try:
                proper.prop_use_fftw()
            except Exception as exc:                 # noqa: BLE001
                return Unsupported(f"prop_use_fftw() failed: {exc}")
        elif config.fft_backend == "mkl":
            mkl_dir = os.environ.get("MKL_DIR")
            if not mkl_dir:
                return Unsupported(
                    "prop_use_ffti() needs MKL_DIR; run scripts/setup_env.sh "
                    "(or use the dragrace-mkl environment)"
                )
            if not os.path.exists(os.path.join(mkl_dir, "libmkl_rt.so")):
                return Unsupported(
                    f"{mkl_dir}/libmkl_rt.so missing -- PROPER builds this path "
                    "with the unversioned soname while conda/PyPI ship only "
                    "libmkl_rt.so.2. scripts/setup_env.sh creates the symlink."
                )
            try:
                proper.prop_use_ffti(MKL_DIR=mkl_dir)
            except Exception as exc:                 # noqa: BLE001
                return Unsupported(f"prop_use_ffti() failed: {exc}")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        return {"array_module": "numpy", "device": "cpu",
                "fft_backend": self._fft_name, "blas": detect_blas()}

    def _build_aperture(self, case: Case, config: Config):
        """prop_hex_wavefront for the segments, prop_rectangular_obscuration
        for the spiders -- PROPER's own tools for a segmented pupil.

        HOW THE OMIT LIST IS DERIVED, because this is the one adapter where it
        could not be read off a public attribute. prop_hex_wavefront walks its
        segments in an inline column scan and exposes no centre for a given
        index, so the obvious route is to transcribe that loop here. This does
        not do that: reproducing a library's traversal by hand is the one place
        a layout would be asserted against a *copy* of the code rather than
        against the code, and it silently rots the moment PROPER reorders
        anything.

        Instead the centres come from PROPER itself. One probe call runs
        prop_hex_wavefront with prop_polygon temporarily replaced by a recorder
        that captures each (xc, yc) and draws nothing, so PROPER performs its own
        traversal and hands back its own ordering. Untimed, and cheap because the
        stand-in returns zeros rather than rasterising 919 hexagons.

        The alternative the brief allowed -- skip the selection entirely and draw
        all 919 segments with a note -- was rejected on its own terms: it is
        15% more segments, which lands directly in the runtime being measured,
        and it also drops the central obscuration, which fails the geometry gate
        (IoU 0.868 against a 0.95 threshold) and so would produce no timing row
        at all.

        GRID. PROPER's beam_ratio makes the pupil fill the grid exactly at
        beam_ratio=1, and its antialiased polygon then indexes one pixel past
        the array on the outermost segments (IndexError from prop_polygon:190).
        The aperture is drawn on N+2 samples at the case's sampling and cropped,
        which keeps dx exactly as the case pins it.
        """
        import numpy as np
        import proper
        from dragrace.apertures import elt_segment_centres, select_for_centres

        seg = case.segmented
        n = case.n_pupil
        grid = n + 2
        state = {
            "case": case, "aperture": True, "gridsize": grid, "crop": (grid - n) // 2,
            "n": n, "diam_m": case.pupil.diameter_m,
            "beam_ratio": n / grid,
            "hexrad": seg.segment_vertex_to_vertex_m / 2.0,   # centre-to-vertex
            "hexsep": seg.segment_spacing_m,                  # centre-to-centre
            "rings": seg.rings,
            "spider_count": seg.spider_count,
            "spider_width": seg.spider_width_m,
            "spider_offset": seg.spider_angle_offset_deg,
        }

        seen: list[tuple[float, float]] = []
        original = proper.prop_polygon

        def _record_only(wf, nvert, radius, xc, yc, **kwargs):
            seen.append((xc, yc))
            return np.zeros((grid, grid))

        proper.prop_polygon = _record_only
        try:
            wf = proper.prop_begin(state["diam_m"], case.wavelength_m, grid,
                                   state["beam_ratio"])
            proper.prop_hex_wavefront(wf, seg.rings, state["hexrad"],
                                      state["hexsep"], NO_APPLY=True)
        finally:
            proper.prop_polygon = original

        centres = np.asarray(seen, dtype=float)
        canonical = elt_segment_centres(case.segmented_spec())
        keep = select_for_centres(centres, canonical, tol=seg.segment_spacing_m * 0.25)
        if len(keep) != seg.n_segments:
            raise ValueError(
                f"PROPER segment selection matched {len(keep)} of "
                f"{seg.n_segments} canonical ELT segments from its own traversal "
                f"of {len(centres)} candidates. Failing rather than drawing the "
                f"wrong telescope."
            )
        state["omit"] = sorted(set(range(len(centres))) - set(keep.tolist()))
        return state

    def build(self, case: Case, config: Config):
        """Almost empty by necessity -- see the module docstring.

        The one real piece of work: embedding the canonical pupil into PROPER's
        grid. PROPER sizes its array from beam_ratio, so the mask is padded to
        gridsize with the aperture occupying N_D samples across, centred.
        """
        if case.is_aperture:
            return self._build_aperture(case, config)

        n = case.n_fft
        off = n // 2 - case.n_pupil // 2
        if case.kind == "plane_to_plane":
            # beam_ratio = beam diameter / grid width; the case's guard band is
            # already in array_samples, so the grid IS the case's grid.
            mask = circular_aperture(case, self.grid_centering)
            opd = opd_waves(case, self.grid_centering)
            return {
                "case": case, "gridsize": n, "free_space": True,
                "passvalue": {
                    "diam_m": case.pupil.diameter_m,
                    "beam_ratio": case.pupil.diameter_m / (case.dx_pupil_m * n),
                    "distance_m": case.propagation.distance_m,
                    "mask": mask,
                    "opd_m": (opd * case.wavelength_m
                              if case.pupil.aberration.coefficients else None),
                },
                "crop": (0, n),
            }
        mask = np.zeros((n, n), dtype=np.float64)
        mask[off:off + case.n_pupil, off:off + case.n_pupil] = circular_aperture(
            case, self.grid_centering)

        opd = opd_waves(case, self.grid_centering)
        opd_m = None
        if case.pupil.aberration.coefficients:
            opd_m = np.zeros((n, n), dtype=np.float64)
            opd_m[off:off + case.n_pupil, off:off + case.n_pupil] = opd * case.wavelength_m

        return {
            "case": case,
            "gridsize": n,
            "passvalue": {
                "diam_m": case.pupil.diameter_m,
                "efl_m": case.output.focal_length_m,
                "beam_ratio": 1.0 / case.q,          # samples per lambda F/D = 1/beam_ratio
                "mask": mask,
                "opd_m": opd_m,
            },
            "crop": (n // 2 - case.n_focus // 2, case.n_focus),
        }

    def propagate(self, state):
        if state.get("aperture"):
            import numpy as np
            import proper

            wf = proper.prop_begin(state["diam_m"], state["case"].wavelength_m,
                                   state["gridsize"], state["beam_ratio"])
            amp = proper.prop_hex_wavefront(
                wf, state["rings"], state["hexrad"], state["hexsep"],
                NO_APPLY=True, OMIT=state["omit"])
            amp = np.asarray(amp[0] if isinstance(amp, tuple) else amp)
            if state["spider_count"]:
                # Six half-rays from the centre. prop_rectangular_obscuration
                # takes a centred rectangle, so each vane is a bar of length D
                # whose centre sits at D/2 along its own direction -- that puts
                # one end at the origin and the other past the pupil edge.
                d = state["diam_m"]
                for i in range(state["spider_count"]):
                    ang = np.deg2rad(state["spider_offset"] + 60.0 * i)
                    proper.prop_rectangular_obscuration(
                        wf, state["spider_width"], d,
                        d / 2.0 * np.cos(ang), d / 2.0 * np.sin(ang),
                        ROTATION=np.degrees(ang) - 90.0)
                amp = amp * np.abs(proper.prop_get_amplitude(wf))
            c, n = state["crop"], state["n"]
            return amp[c:c + n, c:c + n]

        # The prescription is called directly rather than through prop_run().
        # prop_run takes a prescription by NAME and imports it from a file on
        # sys.path, which cannot reach a function defined inside an installed
        # package. Calling it directly exercises the identical propagation code
        # path, minus PROPER's file-based prescription lookup -- and that lookup
        # is per-call overhead a real user does pay, so it is noted in the
        # report rather than quietly excluded. See docs/methodology.md.
        fn = _prescription_free_space if state.get("free_space") else _prescription
        out = fn(state["case"].wavelength_m, state["gridsize"], state["passvalue"])
        if isinstance(out, tuple):
            out = out[0]
        c, npix = state["crop"]
        return np.asarray(out)[c:c + npix, c:c + npix]
