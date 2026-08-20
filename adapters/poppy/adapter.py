"""POPPY adapter, through the API POPPY's own documentation teaches.

STATUS: matrix_dft path exercised against a real install (poppy 1.1.1,
macOS/arm); agrees with the internal reference to roundoff. The fft and
fresnel_tf paths that supports() advertises are NOT implemented -- build()
constructs a detector-plane system regardless of algorithm_class -- so `status`
stays unverified until those are implemented or dropped from supports().

WHAT IS TIMED, AND WHY IT IS calc_psf(). An earlier version of this adapter
called poppy.matrixDFT directly, on the grounds that the MFT is the propagation
the case is about. That measures POPPY's transform kernel, which is not what a
POPPY user runs: the documentation, the tutorials and every example build an
OpticalSystem and call calc_psf(). The gap is not small -- at N_p=1024,
matrixDFT.perform() takes 20.1 ms and calc_psf() 29.5 ms, so 47% of what a user
waits for lives outside the transform, in astropy.units handling, normalisation
and FITS HDU construction. Timing the kernel alone reports a POPPY nobody uses.

So the timed region is one calc_psf() call. Building the OpticalSystem -- the
planes, the pupil optic, the detector -- happens in build() and is untimed,
because it is what any real user hoists out of their loop.

CENTRING. POPPY's OpticalSystem hard-codes MatrixFourierTransform(centering=
'ADJUSTABLE') inside _propagate_mft, and no documented knob reaches it. Its PSF
is therefore centred between the middle pixels, in both planes. That is a
legitimate convention, not an error, so this adapter declares it and the harness
builds the reference and the injected pupil to match (grid_centering below).
Against a pixel-centred reference POPPY scores rel_l2 = 0.28 with the peak one
pixel low; against its own it scores 1.5e-15. Note the corollary: POPPY's
`source_offset_r/theta` can slide the PSF onto a pixel-centred grid, but that
only fixes the focal plane -- the pupil-side half pixel survives as a residual
phase ramp and bottoms out at rel_l2 = 2.3e-2. Declaring the convention is the
honest fix; shimming it is not.

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
from dragrace.grid import circular_aperture, opd_waves, pupil_field


@register("poppy")
class PoppyAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""
    requires = ("poppy", "astropy")

    #: POPPY's OpticalSystem is interpixel-centred and offers no way out of it.
    grid_centering = "interpixel"

    def versions(self) -> dict[str, str]:
        import poppy
        import astropy
        return {"poppy": getattr(poppy, "__version__", "unknown"),
                "astropy": astropy.__version__, "numpy": np.__version__}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.is_aperture:
            if config.is_gpu:
                return Unsupported("GPU config requires CuPy (env dragrace-gpu-cupy)")
            return True
        if case.algorithm_class == "angular_spectrum":
            return Unsupported(
                "POPPY's FresnelWavefront applies the paraxial transfer function "
                "exp(-i pi lambda z f^2) (fresnel._propagate_ptp, Lawrence eq. 87): "
                "it matches an internal_fresnel_tf reference to 1.83e-14 and an "
                "exact angular-spectrum one only to 4.07e-6. Use fresnel_d50_z1m.")
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
        from poppy import accel_math

        self._gpu = config.is_gpu
        self._fft_name = config.fft_backend
        # Toggle names have moved between POPPY versions; set what exists and
        # report what resolved rather than assuming either.
        #
        # use_numexpr is deliberately NOT forced. numexpr is a POPPY accelerator
        # for its elementwise transcendentals, it ships enabled, and no config
        # axis governs it -- the config owns device, FFT, BLAS and threads. An
        # adapter switching off an accelerator the library turns on by default
        # would measure a POPPY nobody runs, which is the same mistake
        # primitive-v1 made (docs/methodology.md). It honours NUMEXPR_NUM_THREADS,
        # which the runner sets from the config, so it costs no thread honesty:
        # measured 3% here, against 44% for the FFT axis below.
        for attr, value in (("use_fftw", config.fft_backend == "pyfftw"),
                            ("use_mkl", config.fft_backend == "mkl"),
                            ("use_cuda", config.is_gpu)):
            if hasattr(poppy.conf, attr):
                try:
                    setattr(poppy.conf, attr, value)
                except Exception:                    # noqa: BLE001
                    pass

        # THIS LINE IS THE WHOLE POINT OF THE BLOCK ABOVE. accel_math snapshots
        # conf.use_fftw / use_numexpr / use_mkl into module-level globals at
        # import time (accel_math.py:69-70) and every FFT dispatch reads the
        # globals, not conf. Setting conf after import therefore changes nothing,
        # and POPPY quietly kept using whatever it resolved on import --
        # pyFFTW, because conf.use_fftw defaults to True and environment.yml
        # installs pyfftw for PROPER. That put pyFFTW-backed POPPY rows on the
        # cpu_numpy_1t control board labelled fft_backend='numpy', worth 1.44x
        # at N=2048 (331 ms against 477 ms), and made cpu_pyfftw_1t and
        # cpu_numpy_1t the same measurement. update_math_settings() is POPPY's
        # own public re-read of conf and is what makes the config axis real.
        try:
            accel_math.update_math_settings()
        except Exception as exc:                     # noqa: BLE001
            return Unsupported(f"poppy.accel_math.update_math_settings() failed: {exc}")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        resolved = {"array_module": "numpy", "device": "cpu",
                    # Read out of accel_math's globals -- the same ones the FFT
                    # dispatch reads -- rather than echoing the request back.
                    # Echoing is what let the pyFFTW leak above reach a plot:
                    # backend.verify can only refuse a mismatch an adapter is
                    # honest enough to report.
                    "fft_backend": self._resolved_fft(), "blas": detect_blas(),
                    # Recorded per result rather than left in the source: the
                    # centring convention is the difference between passing the
                    # gate and failing it by a pixel, and a reader of an old
                    # result should not have to guess which one was in force.
                    "centering": "ADJUSTABLE (poppy_core._propagate_mft, not settable)",
                    "api": "OpticalSystem.calc_psf"}
        try:
            from poppy import accel_math
            resolved["accel_math"] = {
                k: bool(getattr(accel_math, k, False))
                for k in ("_USE_FFTW", "_USE_MKL", "_USE_CUPY", "_USE_NUMEXPR",
                          "_USE_OPENCL")
                if hasattr(accel_math, k)
            }
            if resolved["accel_math"].get("_USE_CUPY"):
                resolved["array_module"] = "cupy"
                resolved["device"] = "cuda"
        except Exception:                            # noqa: BLE001
            pass
        return resolved

    @staticmethod
    def _resolved_fft() -> str:
        """Which FFT POPPY will actually dispatch to, in harness vocabulary.

        Order mirrors the if/elif chain in accel_math.fft_2d, so this reports
        the branch that will really be taken rather than a plausible guess.
        """
        try:
            from poppy import accel_math
        except Exception:                            # noqa: BLE001
            return "unknown"
        for flag, name in (("_USE_CUPY", "native"), ("_USE_OPENCL", "native"),
                           ("_USE_MKL", "mkl"), ("_USE_FFTW", "pyfftw")):
            if getattr(accel_math, flag, False):
                return name
        return "numpy"

    def _build_aperture(self, case: Case, config: Config):
        """MultiHexagonAperture + SecondaryObscuration, composed and sampled.

        POPPY has no ELT, so the pupil is assembled from the generic segmented
        machinery its documentation teaches, and the segment selection is made
        by matching POPPY's own `_aper_center` values against the canonical
        layout rather than by index -- POPPY numbers "clockwise from the segment
        immediately above centre", prysm counts from "up", and lentil follows
        its own hex-ring order, so one index list cannot serve all three.

        The central obscuration is NOT a SecondaryObscuration disc: the ELT's is
        a hexagonal hole formed by leaving 61 segments out, and adding a circular
        stop on top of that would obscure segments the real telescope has.
        SecondaryObscuration is used only for the six spiders, with a zero
        secondary radius.

        Note POPPY's own warning on MultiHexagonAperture: "becomes a bit slow for
        nrings > 4". The ELT needs 17, and that is a finding rather than a
        misuse -- it is the documented way to build this pupil in POPPY.
        """
        import numpy as np
        import astropy.units as u
        import poppy
        from dragrace.apertures import elt_segment_centres, select_for_centres

        seg = case.segmented
        f2f = seg.segment_flat_to_flat_m
        probe = poppy.MultiHexagonAperture(
            rings=seg.rings, flattoflat=f2f * u.m, gap=seg.segment_gap_m * u.m,
            center=True)
        n_all = probe._n_aper_inside_ring(seg.rings + 1)
        # _aper_center documents itself as returning "y, x coords" -- reversed
        # from every other centre convention in this comparison. Taken at face
        # value its lattice looks rotated 30 degrees from the canonical layout
        # (nearest-neighbour directions at 0/+-60/180 rather than +-30/+-90/+-150)
        # and only 180 of the 798 segments match; the reversal is the whole
        # difference, not a real disagreement about where the segments go.
        centres = np.array([probe._aper_center(i) for i in range(n_all)],
                           dtype=float)[:, ::-1]

        canonical = elt_segment_centres(case.segmented_spec())
        keep = select_for_centres(centres, canonical, tol=seg.segment_spacing_m * 0.25)
        if len(keep) != seg.n_segments:
            raise ValueError(
                f"POPPY segment selection matched {len(keep)} of {seg.n_segments} "
                f"canonical ELT segments -- its ring enumeration or _aper_center "
                f"convention has moved. Failing rather than drawing the wrong "
                f"telescope."
            )
        return {"case": case, "aperture": True,
                "segmentlist": [int(i) for i in keep],
                "rings": seg.rings, "f2f": f2f, "gap": seg.segment_gap_m,
                "spider_count": seg.spider_count,
                "spider_width": seg.spider_width_m,
                "spider_offset": seg.spider_angle_offset_deg,
                "npix": case.n_pupil, "grid_size": case.pupil.diameter_m}

    def build(self, case: Case, config: Config):
        """Untimed: the OpticalSystem a user would build once and reuse."""
        import astropy.units as u
        import poppy

        if case.is_aperture:
            return self._build_aperture(case, config)

        if case.kind == "plane_to_plane":
            # FresnelWavefront arrives with its own hard aperture of radius
            # beam_radius already applied. Multiplying the canonical mask into it
            # intersects two discs whose rasterisations differ by a pixel, which
            # cost 3.1e-3 of accuracy; replacing the array outright is what puts
            # the case's aperture -- and only the case's aperture -- into POPPY.
            return {"case": case,
                    "poppy": poppy, "u": u,
                    "beam_radius_m": case.pupil.diameter_m / 2.0,
                    "npix": case.n_across,
                    "oversample": case.n_pupil // case.n_across,
                    "field": pupil_field(case, centering=self.grid_centering
                                         ).astype(np.complex128),
                    "distance_m": case.propagation.distance_m,
                    "free_space": True}

        # The pupil goes in as an ArrayOpticalElement rather than a
        # poppy.CircularAperture: the case pins a hard-edged mask, while
        # CircularAperture antialiases its edge by default. Letting POPPY
        # rasterise its own aperture would put a rasterisation difference --
        # which costs real time, and which each of these codes makes
        # differently -- inside a propagation comparison. See docs/conventions.md.
        transmission = circular_aperture(case, self.grid_centering)
        opd_m = opd_waves(case, self.grid_centering) * case.wavelength_m

        # POPPY sizes the input wavefront from npix and the pupil diameter,
        # where the diameter is the *array* extent, not the aperture's.
        array_extent_m = case.pupil.diameter_m * case.n_pupil / case.n_across

        # Detector pixels are angular. One case sample is (lambda/D)/q radians;
        # POPPY wants arcsec. fov_pixels then fixes N_f exactly, and
        # oversample=1 keeps the computed grid the detector grid -- POPPY's
        # default of 2 would compute at twice the sampling and bin down, which
        # is a different (and more expensive) calculation than the case asks for.
        lam_over_d_rad = case.wavelength_m / case.pupil.diameter_m
        pixelscale_arcsec = np.degrees(lam_over_d_rad / case.q) * 3600.0

        osys = poppy.OpticalSystem(npix=case.n_pupil, oversample=1,
                                   pupil_diameter=array_extent_m * u.m)
        osys.add_pupil(poppy.ArrayOpticalElement(
            transmission=transmission, opd=opd_m,
            pixelscale=(case.dx_pupil_m * u.m / u.pixel)))
        osys.add_detector(pixelscale=pixelscale_arcsec, fov_pixels=case.n_focus)

        return {"case": case, "osys": osys, "wavelength": case.wavelength_m * u.m}

    def propagate(self, state):
        """Exactly the call POPPY's documentation puts in front of a user.

        Free-space branch: propagate_fresnel mutates the wavefront and advances
        its z, so a second PSF needs a second wavefront -- constructing it is
        per-call work POPPY gives no way to hoist, exactly as with lentil.

        `normalize` is left at its default: a user who does not pass it gets
        'first', and its cost is part of what calc_psf costs. The harness fits
        out the resulting normalisation anyway (validate.compare reports it as
        scale_abs), so nothing is lost by leaving POPPY's default in place.
        """
        if state.get("aperture"):
            import astropy.units as u
            import poppy

            segs = poppy.MultiHexagonAperture(
                rings=state["rings"], flattoflat=state["f2f"] * u.m,
                gap=state["gap"] * u.m, center=True,
                segmentlist=state["segmentlist"])
            optics = [segs]
            if state["spider_count"]:
                optics.append(poppy.SecondaryObscuration(
                    secondary_radius=0.0 * u.m,
                    n_supports=state["spider_count"],
                    support_width=state["spider_width"] * u.m,
                    support_angle_offset=state["spider_offset"]))
            compound = poppy.CompoundAnalyticOptic(opticslist=optics, name="ELT")
            return compound.sample(npix=state["npix"],
                                   grid_size=state["grid_size"] * u.m)

        if state.get("free_space"):
            poppy, u = state["poppy"], state["u"]
            wf = poppy.FresnelWavefront(
                state["beam_radius_m"] * u.m,
                wavelength=state["case"].wavelength_m * u.m,
                npix=state["npix"], oversample=state["oversample"])
            wf.wavefront = state["field"].copy()
            wf.propagate_fresnel(state["distance_m"] * u.m)
            return wf
        return state["osys"].calc_psf(wavelength=state["wavelength"])

    def sync(self, result) -> None:
        if self._gpu:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except ImportError:
                pass

    def to_host(self, result) -> np.ndarray:
        """The intensity PSF, which is what calc_psf hands a user.

        Free-space branch returns a FresnelWavefront, whose .wavefront is the
        complex field -- so no complex_field() override is needed there.

        Extension 0 is the computed (here also detector-sampled, since
        oversample=1) PSF. Timed as the device->host cost like any other
        adapter's; the accuracy gate reads complex_field() instead.
        """
        if hasattr(result, "wavefront"):
            return np.asarray(result.wavefront)
        if isinstance(result, np.ndarray):        # aperture board: a plain mask
            return result
        return np.asarray(result[0].data)

    def complex_field(self, state, result) -> np.ndarray:
        if state.get("aperture"):
            # A drawn pupil is already the quantity being gated; there is no
            # field to recover and nothing extra to pay for.
            return self.to_host(result)
        if state.get("free_space"):
            return np.asarray(result.wavefront).astype(state["case"].dtype)
        return self._complex_field_calc_psf(state, result)

    def _complex_field_calc_psf(self, state, result) -> np.ndarray:
        """Untimed re-run with return_final=True, for the gate only.

        POPPY documents return_final as the way to get the complex PSF "without
        the memory usage of return_intermediates". It is the same propagation
        through the same OpticalSystem, but it costs ~15% more than the plain
        call (34.0 ms vs 29.5 ms at N_p=1024), so it must not be what the clock
        sees -- hence a separate call here rather than folding it into
        propagate().
        """
        _, final = state["osys"].calc_psf(wavelength=state["wavelength"],
                                          return_final=True)
        wf = final[0] if isinstance(final, (list, tuple)) else final
        return np.asarray(wf.wavefront).astype(state["case"].dtype)
