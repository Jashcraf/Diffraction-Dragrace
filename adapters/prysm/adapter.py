"""prysm adapter -- forward board and gradient board.

API verified against the prysm working checkout (v0.22, branch
fix_lbfgsb_divide_cupy) on 2026-08-05:

    prysm.propagation.prepare_executor(pupil_dx, pupil_samples, focal_dx,
                                       focal_samples, wavelength, efl,
                                       focal_shift=(0,0), kind='mdft'|'czt')
    prysm.propagation.focus_dft(field, executor)        -> executor(field)
    prysm.propagation.focus_dft_adjoint(grad, executor) -> executor.adjoint(grad)
    prysm.propagation.focus(field, Q) / focus_adjoint(grad, Q)   [FFT path]
    prysm.polynomials.sum_of_2d_modes(modes, weights)
    prysm.polynomials.sum_of_2d_modes_adjoint(modes, databar)
    prysm.mathops.np / fft            BackendShim, swap _srcmodule for CuPy
    prysm.fttools.fftrange(n) = arange(-(n//2), -(n//2)+n)

That last line matters: prysm's grid convention is identical to the harness's
(centre at index n//2), so no re-centring shim is needed and the two agree to
roundoff rather than to half a pixel.

prysm exposes `kind='czt'` alongside 'mdft'. That makes the chirp-Z algorithm
class -- the theoretical floor for arbitrary output sampling -- directly
measurable here, which is not true of the other five codes.

NOTE ON UNITS: prysm mixes mm (pupil spacing, efl) with microns (focal spacing,
wavelength). The conversions below are the whole reason this adapter exists;
getting them wrong produces a plausible-looking PSF at the wrong scale, which
the accuracy gate catches as a large rel_l2 with scale_abs far from 1.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import circular_aperture, gradient_parameters, opd_waves, pupil_field


@register("prysm")
class PrysmAdapter(Adapter):
    status = "unverified"        # API read from source; not yet run against an install
    reviewed_by = ""             # invite Brandon Dube before publishing results
    requires = ("prysm",)

    def versions(self) -> dict[str, str]:
        import prysm
        return {"prysm": getattr(prysm, "__version__", "unknown"), "numpy": np.__version__}

    #: prysm ships a reverse-mode API -- sum_of_2d_modes_adjoint and
    #: focus_dft_adjoint -- so the forward model returns (loss, dloss/dtheta)
    #: and the optimiser never forms a difference quotient.
    retrieval_gradient = "analytic"

    #: The whole differentiated chain is prysm's backend shim, so pointing
    #: mathops.np at CuPy moves it to the device without touching this adapter.
    retrieval_devices = ("cpu", "gpu")

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.is_retrieval:
            sup = self.retrieval_support(case, config)
            return sup if not sup else self.supports_gradient()
        if case.is_aperture:
            if config.is_gpu:
                return Unsupported("GPU config requires CuPy (env dragrace-gpu-cupy)")
            return True
        if case.algorithm_class == "angular_spectrum":
            return Unsupported(
                "prysm's Wavefront.free_space applies the paraxial Fresnel transfer "
                "function, not the exact sqrt form -- it matches an internal_fresnel_tf "
                "reference to 1.8e-14 and an exact angular-spectrum one only to 4.07e-6. "
                "Use a fresnel_tf case (fresnel_d50_z1m).")
        if case.algorithm_class not in ("matrix_dft", "fft", "czt", "fresnel_tf"):
            return Unsupported(f"no prysm path for {case.algorithm_class}")
        if config.is_gpu:
            try:
                import cupy  # noqa: F401
            except ImportError:
                return Unsupported("GPU config requires CuPy (env dragrace-gpu-cupy)")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        from prysm import mathops
        from prysm.conf import config as pconf

        self._gpu = config.is_gpu
        if config.is_gpu:
            import cupy as cp
            import cupyx.scipy.fft as cpfft
            import cupyx.scipy.ndimage as cpndi
            mathops.np._srcmodule = cp
            mathops.fft._srcmodule = cpfft
            mathops.ndimage._srcmodule = cpndi
        elif config.fft_backend == "xla":
            # prysm's backend shim is not CuPy-specific: jax.numpy satisfies the
            # same array protocol, so pointing the array module at it runs the
            # whole propagation through XLA with no change to prysm itself.
            # Verified against the internal reference at rel_l2 = 1.3e-15.
            #
            # Not jitted, deliberately. prysm's propagation is written for eager
            # array APIs; wrapping it in jax.jit would trace through prysm's
            # Python and measure a different thing from what a prysm user gets
            # by swapping the backend, which is the documented way to use it.
            try:
                import jax.numpy as jnp
                import jax.scipy.ndimage as jndi
            except ImportError as exc:
                return Unsupported(f"XLA config requires JAX ({exc})")
            mathops.np._srcmodule = jnp
            mathops.ndimage._srcmodule = jndi
            try:                       # jax.scipy.fft is partial; leave scipy's
                import jax.scipy.fft as jfft          # noqa: F401
                mathops.fft._srcmodule = jfft
            except ImportError:
                import scipy.fft as _sfft
                mathops.fft._srcmodule = _sfft
        else:
            import numpy as _np
            import scipy.fft as _sfft
            import scipy.ndimage as _sndi
            mathops.np._srcmodule = _np
            mathops.ndimage._srcmodule = _sndi
            if config.fft_backend == "mkl":
                import mkl_fft._scipy_fft as mfft
                mathops.fft._srcmodule = mfft
            elif config.fft_backend == "pyfftw":
                import pyfftw.interfaces.scipy_fft as pfft
                import pyfftw
                pyfftw.interfaces.cache.enable()
                mathops.fft._srcmodule = pfft
            elif config.fft_backend == "numpy":
                # prysm's own default is scipy.fft, and leaving it there on a
                # config that asks for numpy is not a neutral choice: the two are
                # different pocketfft builds, and the whole point of cpu_numpy_1t
                # is that every adapter transforms through the same one. Pointing
                # the shim is prysm's documented mechanism for exactly this --
                # mathops.set_fft_backend_to_mkl_fft() does the same thing one
                # line over -- so this honours the config axis without deviating
                # from how prysm is meant to be used. fttools.next_fast_len and
                # fttools.fftfreq both carry AttributeError fallbacks for
                # backends that lack them, so numpy.fft is a complete substitute
                # on this path.
                mathops.fft._srcmodule = _np.fft
            else:                                    # scipy_pocketfft, prysm's default
                mathops.fft._srcmodule = _sfft

        pconf.precision = 32 if config.precision_override == "complex64" else 64
        self._fft_name = config.fft_backend
        # Set per configure() rather than as a class attribute: prysm honours
        # the BLAS and thread axes perfectly well on NumPy, and only loses them
        # when its array module is pointed at XLA.
        self.config_axes_not_selectable = ("blas", "threads") if config.fft_backend == "xla" else ()
        return True

    #: Module name of prysm's FFT shim -> the harness's backend vocabulary.
    #: Longest match first, so 'mkl_fft._scipy_fft' cannot be read as scipy.
    _FFT_MODULE_NAMES = (
        ("mkl_fft", "mkl"),
        ("pyfftw", "pyfftw"),
        ("jax", "xla"),
        ("cupyx", "native"),
        ("numpy.fft", "numpy"),
        ("scipy.fft", "scipy_pocketfft"),
    )

    def resolve_backend(self) -> dict:
        from prysm import mathops
        from dragrace.backend import detect_blas
        arr = getattr(mathops.np._srcmodule, "__name__", "?")
        on_xla = "jax" in arr
        fft_mod = getattr(mathops.fft._srcmodule, "__name__", "?")
        # Read back what the shim actually points at rather than echoing the
        # request. Reporting the requested name is how a backend leak survives
        # into a plotted result: the guard in dragrace.backend.verify can only
        # catch a mismatch that an adapter is honest enough to report.
        resolved_fft = next((v for k, v in self._FFT_MODULE_NAMES if fft_mod.startswith(k)),
                            fft_mod)
        return {
            "array_module": arr,
            "fft_module": fft_mod,
            "fft_backend": resolved_fft,
            "device": "cuda" if "cupy" in arr else "cpu",
            # On XLA there is no BLAS in the path -- XLA emits its own kernels,
            # so naming one would be a label with nothing behind it.
            "blas": "unknown" if (self._gpu or on_xla) else detect_blas(),
        }

    # ------------------------------------------------------------ lifecycle --
    def _build_aperture(self, case: Case, config: Config):
        """prysm's CompositeHexagonalAperture, plus spiders from prysm.geometry.

        prysm has no ELT, so the pupil is composed the way its documentation
        teaches: lay out a hexagonal composite and exclude the segments you do
        not want. The exclude list is derived by matching prysm's own segment
        centres against the canonical layout rather than by counting rings --
        prysm, POPPY and lentil each number their segments differently, and an
        index list written for one silently draws a different telescope in
        another.

        CompositeHexagonalAperture takes flat-to-flat, so the case's
        vertex-to-vertex 1.45 m is converted here; getting that backwards is the
        single easiest way to build a convincing pupil of the wrong size.
        """
        import numpy as np
        from prysm.coordinates import make_xy_grid
        from prysm.segmented import CompositeHexagonalAperture
        from dragrace.apertures import elt_segment_centres, select_for_centres

        seg = case.segmented
        f2f = seg.segment_flat_to_flat_m
        x, y = make_xy_grid(case.n_pupil, diameter=case.pupil.diameter_m)

        # Read prysm's own segment centres out of an unexcluded composite rather
        # than reimplementing its lattice: hex_to_xy's radius argument and
        # hex_ring's "roll so the first element is north" are exactly the kind of
        # convention that a reimplementation gets subtly wrong and that then
        # shows up as a plausible pupil with the wrong segments missing. This
        # construction is untimed -- it is the once-per-geometry work a user does
        # to find their exclude list, which is what build() is for.
        probe = CompositeHexagonalAperture(x, y, seg.rings, f2f, seg.segment_gap_m, 90)
        centres = np.asarray(probe.all_centers, dtype=float)
        spacing = seg.segment_spacing_m

        canonical = elt_segment_centres(case.segmented_spec())
        keep = select_for_centres(centres, canonical, tol=spacing * 0.25)
        exclude = tuple(sorted(set(range(len(centres))) - set(keep.tolist())))
        if len(keep) != seg.n_segments:
            raise ValueError(
                f"prysm segment selection matched {len(keep)} of {seg.n_segments} "
                f"canonical ELT segments. prysm's ring enumeration or its "
                f"hex_to_xy convention has moved; the exclude list would draw the "
                f"wrong telescope, so this fails rather than producing a plot."
            )

        return {"case": case, "aperture": True, "x": x, "y": y,
                "rings": seg.rings, "f2f": f2f, "gap": seg.segment_gap_m,
                "exclude": exclude,
                "spider_width": seg.spider_width_m,
                "spider_count": seg.spider_count,
                "spider_offset": seg.spider_angle_offset_deg}

    def _build_retrieval(self, case: Case, config: Config):
        """Untimed: the executor, the basis, the mask, and the observed PSF.

        The differentiated chain is prysm's own, called forwards and then
        backwards. Every step has a prysm partner except the two that are pure
        arithmetic -- the phasor and |.|^2 -- which are written out here, the
        same division the gradient board makes:

            phs  = sum_of_2d_modes(basis, theta)   <-> sum_of_2d_modes_adjoint
            W    = amp * exp(2i.pi.phs)            <-> phsbar = -4.pi.Im(conj(Wbar).W)
            E    = focus_dft(W, executor)          <-> focus_dft_adjoint
            I    = |E|^2                           <-> Ebar = Ibar.E
            L    = mean(((I - I_obs)/s)^2)         <-> Ibar = 2.resid/(n.s)

        WHICH WIRTINGER CONVENTION THIS CHAIN IS IN, because there are two and
        mixing them is silent. For a real loss of a complex variable both of
        these are self-consistent, and they are complex conjugates of each other:

            holomorphic   track dL/dz   -> a linear map backpropagates as A^T
            conjugate     track dL/dz*  -> the same map backpropagates as A^H

        prysm's API is the second: executor.adjoint applies the conjugate
        transpose (fttools.MDFT.adjoint: `Ey.conj().T @ grad @ Ex.conj()`),
        which is the true Wirtinger adjoint dL/dW* = A^H(dL/dE*). So it must be
        handed dL/dE* = Ibar.E -- with E, not conj(E), since I = E.conj(E) gives
        dI/dE* = E -- and the conjugation taken at the phasor step instead.
        `adapters/numpy_baseline` is in the first convention throughout: it
        forms dL/dE = Ibar.conj(E) and uses a plain transpose. NEITHER IS
        WRONG. Measured on this case, the two chains agree with central
        differences at 4.43e-8 apiece, their intermediate cotangents are exact
        conjugates of one another, and their parameter gradients are
        bit-identical.

        What is wrong is crossing the seam. An earlier version of this method
        took numpy_baseline's Ebar and handed it to prysm's A^H, which
        conjugates twice: that gradient is off by up to 68x per component. It
        does not raise, and because L-BFGS-B partly absorbs a bad gradient into
        its step length the symptom is not an error but a STALL -- 1 iteration
        and 44 function evaluations, a line search failing over and over --
        which on a timing board would have read as "prysm is slow".
        docs/gradient_board.md already states the rule this broke: the
        intermediate complex cotangents differ between codes by a conjugation,
        so never transcribe an intermediate from one code's chain into
        another's. The finite-difference check in tests/test_retrieval.py is
        what pins it.

        The 1/s from the loss normalisation appears twice in Ibar: once for the
        residual and once for the intensity that residual is differentiated
        against. Dropping one gives a gradient wrong by a constant factor, which
        L-BFGS-B partly absorbs into its step length -- so it still converges,
        just more slowly, and the board would report prysm as needing more
        iterations than it does. That failure is invisible without a
        finite-difference check, which is why one is in tests/.

        No tape and no tracing: the adapter chooses explicitly which forward
        intermediates to keep (W and E), so the memory is minimal and
        predictable, unlike an XLA-decided one. Expected primitive count is 2
        forward GEMMs and 2 adjoint GEMMs per evaluation; 6 would mean the chain
        is wrong and would unfairly penalise prysm.
        """
        from prysm.propagation import focus_dft, prepare_executor

        from dragrace.grid import aperture_mask
        from dragrace.retrieval import loss_scale, retrieval_parameters

        # WHERE THE HOST/DEVICE BOUNDARY IS, on the GPU configs. L-BFGS-B is
        # scipy's and runs on the host, so theta arrives as a host array and the
        # (loss, gradient) pair has to go back as host scalars -- 11 doubles up,
        # 1 + 11 doubles down, per evaluation. That transfer is charged: it is
        # what a GPU retrieval driven by scipy actually costs, and hiding it
        # would price a loop nobody can run. Everything between those two points
        # -- the basis, the mask, the observed PSF, the phasor, both DFTs --
        # stays on the device, so the O(N^2) arrays never cross.
        xp = np
        if self._gpu:
            import cupy as xp                                        # noqa: N813

        _, theta_true, theta_init, basis = retrieval_parameters(
            case, self.grid_centering)
        basis = xp.asarray(basis)
        amp = xp.asarray(aperture_mask(case, self.grid_centering))
        executor = prepare_executor(
            pupil_dx=case.dx_pupil_m * 1e3,          # m -> mm
            pupil_samples=case.n_pupil,
            focal_dx=case.dx_focus_m * 1e6,          # m -> um
            focal_samples=case.n_focus,
            wavelength=case.wavelength_m * 1e6,      # m -> um
            efl=case.output.focal_length_m * 1e3,    # m -> mm
            kind="mdft",
        )

        def field(theta):
            from prysm.polynomials import sum_of_2d_modes
            phs = sum_of_2d_modes(basis, xp.asarray(theta, dtype=float))
            w = amp * xp.exp(2j * xp.pi * phs)
            return w, focus_dft(w, executor)

        observed = xp.abs(field(theta_true)[1]) ** 2
        s = float(loss_scale(self._host(observed)))

        def loss_and_grad(theta):
            from prysm.polynomials import sum_of_2d_modes_adjoint
            from prysm.propagation import focus_dft_adjoint

            w, e = field(theta)
            resid = (xp.abs(e) ** 2 - observed) / s
            ibar = 2.0 * resid / (resid.size * s)
            ebar = ibar * e                          # dL/dE*, since dI/dE* = E
            wbar = focus_dft_adjoint(ebar, executor)  # prysm applies A^H
            phsbar = -4.0 * xp.pi * xp.imag(xp.conj(wbar) * w)
            grad = sum_of_2d_modes_adjoint(basis, phsbar)
            # float() and _host() are each a device sync. They are also the only
            # two the loop needs, and they are unavoidable: scipy cannot test a
            # device scalar for convergence.
            return float(xp.mean(resid ** 2)), self._host(xp.real(grad))

        return {"case": case, "retrieval": True, "jac": True, "fun": loss_and_grad,
                "psf": lambda th: self._host(xp.abs(field(th)[1]) ** 2),
                "theta0": theta_init,
                "loss_initial": loss_and_grad(theta_init)[0]}

    @staticmethod
    def _host(a):
        """Device array -> host NumPy, and a no-op on CPU."""
        return np.asarray(a.get() if hasattr(a, "get") else a)

    def retrieval_psf(self, state, theta) -> np.ndarray:
        return np.asarray(state["psf"](theta))

    def retrieval_report(self, state, result) -> dict:
        from dragrace.retrieval import make_report
        return make_report(
            result, state["loss_initial"], state["case"],
            state["case"].n_focus ** 2,
            forward_model="prysm focus_dft + focus_dft_adjoint (hand-written adjoint)")

    def build(self, case: Case, config: Config):
        if case.is_retrieval:
            return self._build_retrieval(case, config)

        if case.is_aperture:
            return self._build_aperture(case, config)

        from prysm.propagation import Wavefront, prepare_executor

        field = pupil_field(case, centering=self.grid_centering)
        if self._gpu:
            import cupy as cp
            field = cp.asarray(field)
        elif self._fft_name == "xla":
            import jax
            import jax.numpy as jnp
            field = jax.device_put(jnp.asarray(field))

        # The Wavefront object, not the bare array: prysm's documentation,
        # tutorials and examples all propagate a Wavefront, and it is what
        # carries dx and wavelength so a user cannot silently mismatch them.
        # Constructing it is untimed here for the same reason the OpticalSystem
        # is untimed for POPPY -- it is hoisted out of any real loop. Measured
        # cost of the wrapper at N_p=1024: 16.5 ms via Wavefront.focus_dft
        # against 16.8 ms for the module-level focus_dft on a raw array, i.e.
        # none. prysm's user-facing layer is genuinely free.
        wf = Wavefront(field, case.wavelength_m * 1e6,      # m -> um
                       case.dx_pupil_m * 1e3,               # m -> mm
                       space="pupil")

        state = {"case": case, "field": field, "wf": wf, "cls": case.algorithm_class}
        if case.algorithm_class == "fresnel_tf":
            # Q=1: no padding. The case supplies its own guard band, and prysm's
            # default Q=2 would double the transform size to prevent a wraparound
            # that cannot happen here.
            #
            # THE TRANSFER FUNCTION IS DELIBERATELY *NOT* HOISTED, even though
            # prysm lets you: free_space(tf=...) accepts a precomputed kernel and
            # angular_spectrum_transfer_function's docstring opens "Precompute
            # the transfer function of free space". Passing it would be bit-
            # identical and ~7% faster here, and it is still the wrong call to
            # time. prysm's own tutorial (docs/source/tutorials/Double-Slit
            # Experiment.ipynb) teaches `wf.free_space(D)`; `tf=` appears only in
            # a unit test. This board measures how each library is *meant* to be
            # driven against what that costs, so a per-call kernel rebuild that
            # the documented path performs is part of prysm's number, exactly as
            # POPPY is charged for rebuilding its FresnelWavefront and PROPER for
            # re-executing its whole prescription. Hoisting here would flatter
            # prysm for an optimisation its users are not taught to reach for.
            state["dz_mm"] = case.propagation.distance_m * 1e3
            return state
        if case.algorithm_class in ("matrix_dft", "czt"):
            state["executor"] = prepare_executor(
                pupil_dx=case.dx_pupil_m * 1e3,          # m -> mm
                pupil_samples=case.n_pupil,
                focal_dx=case.dx_focus_m * 1e6,          # m -> um
                focal_samples=case.n_focus,
                wavelength=case.wavelength_m * 1e6,      # m -> um
                efl=case.output.focal_length_m * 1e3,    # m -> mm
                kind="czt" if case.algorithm_class == "czt" else "mdft",
            )
        else:
            state["Q"] = case.q
            n_pad = case.n_fft
            state["crop"] = (n_pad // 2 - case.n_focus // 2, case.n_focus)
        return state

    def propagate(self, state):
        """Wavefront methods, which is how prysm documents propagation."""
        if state.get("retrieval"):
            from dragrace.retrieval import minimise
            return minimise(state["fun"], state["theta0"], state["case"],
                            jac=state["jac"])

        if state.get("aperture"):
            from prysm.geometry import spider
            from prysm.segmented import CompositeHexagonalAperture

            cha = CompositeHexagonalAperture(
                state["x"], state["y"], state["rings"], state["f2f"],
                state["gap"], 90, state["exclude"])
            amp = cha.amp
            if state["spider_count"]:
                # prysm.geometry.spider returns the VANES (True where obscured),
                # not the transmission -- measured fill 0.033 for six 0.4 m vanes
                # on a 39 m pupil. Multiplying by it directly keeps only the
                # spider, which is a mistake that produces a pupil so obviously
                # wrong it fails the gate instantly rather than quietly.
                vanes = spider(state["spider_count"], state["spider_width"],
                               state["x"], state["y"],
                               rotation=state["spider_offset"])
                amp = amp * (1.0 - np.asarray(vanes, dtype=float))
            return amp

        wf = state["wf"]
        if state["cls"] == "fresnel_tf":
            return wf.free_space(dz=state["dz_mm"], Q=1)
        if state["cls"] in ("matrix_dft", "czt"):
            return wf.focus_dft(state["executor"])
        out = wf.focus(state["case"].output.focal_length_m * 1e3, state["Q"])
        c, n = state["crop"]
        return out.data[c:c + n, c:c + n]

    def sync(self, result) -> None:
        if self._gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        elif self._fft_name == "xla":
            # JAX dispatches asynchronously even on CPU. Without this the clock
            # stops before any arithmetic has happened: the first XLA-backed run
            # of this adapter reported 0.045 ms for 1.208 GFLOP -- 26 TFLOP/s on
            # a laptop CPU -- which is dispatch latency, not a record.
            import jax
            jax.block_until_ready(result.data if hasattr(result, "data") else result)

    def to_host(self, result) -> np.ndarray:
        # focus_dft returns a Wavefront; the FFT path has already cropped to a
        # bare array. Both carry the complex field, so no separate
        # complex_field() override is needed. A retrieval returns an Outcome,
        # whose deliverable is its coefficients.
        if hasattr(result, "theta"):
            return np.asarray(result.theta)
        arr = result.data if hasattr(result, "data") else result
        if self._gpu:
            import cupy as cp
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def device_memory(self):
        if not self._gpu:
            return None
        import cupy as cp
        return int(cp.get_default_memory_pool().total_bytes())

    # ------------------------------------------------------ gradient board --
    def supports_gradient(self) -> bool | Unsupported:
        try:
            from prysm.propagation import focus_dft_adjoint  # noqa: F401
            from prysm.polynomials import sum_of_2d_modes_adjoint  # noqa: F401
        except ImportError as exc:
            return Unsupported(
                f"prysm adjoint API missing ({exc}). PyPI tops out at 0.21.1; "
                "install from git or use scripts/setup_env.sh --local-prysm"
            )
        return True

    def build_gradient(self, case: Case, config: Config):
        from prysm.propagation import prepare_executor

        noll, theta0, basis = gradient_parameters(case)
        amp = circular_aperture(case)
        executor = prepare_executor(
            pupil_dx=case.dx_pupil_m * 1e3, pupil_samples=case.n_pupil,
            focal_dx=case.dx_focus_m * 1e6, focal_samples=case.n_focus,
            wavelength=case.wavelength_m * 1e6, efl=case.output.focal_length_m * 1e3,
            kind="mdft",
        )
        xp = np
        if self._gpu:
            import cupy as xp                                        # noqa: N813
        basis_d = xp.asarray(basis)
        amp_d = xp.asarray(amp)

        # Unaberrated target PSF, on the same executor so normalisation matches.
        from prysm.propagation import focus_dft
        e0 = focus_dft(amp_d.astype(xp.complex128), executor)
        return {
            "case": case, "executor": executor, "basis": basis_d, "amp": amp_d,
            "theta": theta0, "theta_d": xp.asarray(theta0),
            "target": xp.abs(e0) ** 2, "xp": xp,
            "static_opd": xp.asarray(opd_waves(case)),
        }

    def gradient(self, state):
        """prysm's own reverse-mode chain, called in reverse order.

        Each step below has a prysm partner except the two that are pure
        arithmetic (the phasor and |.|^2), which are written out. This is
        manual reverse mode: no tape, no tracing, and the adapter chooses
        explicitly which forward intermediates to retain -- so the memory
        profile is minimal and predictable, unlike an XLA-decided one.

        Expected primitive count: 2 forward GEMMs + 2 adjoint GEMMs. The ledger
        asserts this; a chain that issues 6 is wrong and would unfairly
        penalise prysm on the board.
        """
        from prysm.polynomials import sum_of_2d_modes, sum_of_2d_modes_adjoint
        from prysm.propagation import focus_dft, focus_dft_adjoint

        xp = state["xp"]
        basis, amp, theta = state["basis"], state["amp"], state["theta_d"]
        executor, target = state["executor"], state["target"]

        # ---- forward -------------------------------------------------------
        phs = sum_of_2d_modes(basis, theta) + state["static_opd"]
        w = amp * xp.exp(2j * xp.pi * phs)
        e = focus_dft(w, executor)
        inten = xp.abs(e) ** 2
        resid = inten - target
        loss = float(xp.mean(resid ** 2))

        # ---- reverse -------------------------------------------------------
        # prysm's executor.adjoint is the conjugate transpose, so this chain is
        # in the dL/dz* convention: it must be handed dL/dE* = Ibar*E -- with E,
        # because I = E.conj(E) gives dI/dE* = E -- and the conjugation taken at
        # the phasor step. numpy_baseline is in the dL/dz convention instead,
        # pre-conjugating into Ebar and using a plain transpose. Both are
        # correct and agree bit-for-bit on the parameter gradient; crossing the
        # seam conjugates twice, for a gradient wrong by up to 68x per
        # component. See _build_retrieval for the measurement.
        ibar = 2.0 * resid / resid.size
        ebar = ibar * e                             # dL/dE*, since dI/dE* = E
        wbar = focus_dft_adjoint(ebar, executor)    # A^H, the MDFT's adjoint
        phsbar = -4.0 * xp.pi * xp.imag(xp.conj(wbar) * w)   # W = amp*exp(2i.pi.phs)
        grad = sum_of_2d_modes_adjoint(basis, phsbar)
        return loss, xp.asarray(grad).real

    def gradient_theta(self, state) -> np.ndarray:
        return np.asarray(state["theta"])
