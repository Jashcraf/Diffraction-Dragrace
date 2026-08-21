"""dLux adapter (JAX/XLA) -- forward board and gradient board.

STATUS: unverified. dLux is not installed on the machine this was written on,
so the dLux-specific model construction below is written from its documented
API and must be checked against an install before any result is published.
`dragrace doctor` reports this status so an untested adapter is never mistaken
for a measured one.

The JAX-specific machinery, however, is the point of this file and is what the
rest of the suite is built around:

ASYNCHRONOUS DISPATCH. jnp operations return immediately with an array that may
not be computed. sync() calls jax.block_until_ready over every leaf of the
returned pytree, inside the clock. Without it propagate() returns before any
arithmetic has happened. The harness's sync-scaling guard exists to catch a
sync that silently stops blocking.

AHEAD-OF-TIME COMPILATION. build() lowers and compiles explicitly:

    compiled = jax.jit(fn).lower(*args).compile()

so the timed region is pure execution with no tracing ambiguity, and compile
time is measured as a first-class number rather than hidden inside a warm-up.
This also unlocks two things nothing else in the suite offers:

    compiled.cost_analysis()    exact FLOP count, straight from XLA
    compiled.memory_analysis()  exact buffer sizes: temps, args, outputs

The FLOP count is a hardware-independent efficiency metric, and on the gradient
board it gives flops(grad)/flops(forward) measured rather than assumed -- the
board's headline question, "does autodiff cost more arithmetic than a
hand-written adjoint for this physics?".

x64. JAX defaults to float32/complex64. JAX_ENABLE_X64 must be set before the
first jax import (scripts/setup_env.sh writes it into activate.d). The harness
asserts the realised dtype against the case and fails on a mismatch.

MEMORY. RSS is meaningless for JAX. With XLA_PYTHON_CLIENT_PREALLOCATE=false,
device_memory() reads peak_bytes_in_use from the device's memory_stats().
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import (circular_aperture, focus_coords, gradient_parameters,
                           opd_waves, pupil_coords)


@register("dlux")
class DLuxAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""             # invite Louis Desdoigts before publishing results
    requires = ("jax", "dLux")

    #: dLux's AngularOpticalSystem centres its PSF between the middle pixels,
    #: like POPPY. Measured: rel_l2 3.6e-15 against an interpixel reference,
    #: 0.28 against a pixel-centred one.
    grid_centering = "interpixel"

    #: XLA supplies its own kernels: there is no FFT library to select and no
    #: BLAS to point at. Declared so the harness records a run on a `fft=numpy`
    #: config as "the FFT axis did not apply here" rather than refusing it as a
    #: mislabel or, worse, letting it pass unremarked. dLux belongs on the board
    #: -- XLA is not an alternative backend for it, it is the only one -- but a
    #: dLux row is never a data point in a backend comparison.
    #: threads is NOT in this list any more, and the story of why it was is the
    #: reason the harness now measures cpu/wall on every timed region.
    #:
    #: It used to be declared unselectable on the grounds that "the XLA_FLAGS
    #: knobs produced identical timings at 1 and 8 threads when tested". The
    #: timings were indeed identical -- because the knobs did nothing, and XLA
    #: used every core at both settings. Measured on jaxlib 0.10.2 at N_p=1024,
    #: cpu/wall = 10.06 with no XLA_FLAGS, 10.04 with
    #: --xla_cpu_multi_thread_eigen=false, 10.05 with the thunk runtime
    #: disabled: ~10 cores throughout, on boards labelled threads=1. On the
    #: phase-retrieval board that reversed dLux's standing against prysm --
    #: 877 ms against 1353 unpinned, 2117 against 1737 with both held to one
    #: core. "Cannot verify" had quietly become "is not true".
    #:
    #: dragrace.worker._pin_cpus now enforces the axis with CPU affinity, which
    #: is a property of the process that XLA cannot opt out of, and
    #: metrics.Timing records the realised core count so the claim is checked
    #: rather than asserted. The axis is therefore honoured, and saying
    #: otherwise would put a false caveat on a figure that no longer needs one.
    config_axes_not_selectable = ("fft", "blas")   # refined in configure()

    #: Measured peak-memory model for the aperture board, and it had to be
    #: measured. The obvious estimate -- one float64 per segment per pixel,
    #: n_segments x N^2 x 8 -- understates the real peak by more than an order of
    #: magnitude, because XLA does not simply stack and reduce: soft_reg_polygon
    #: evaluates a signed distance against all six edges of every hexagon, and
    #: the fused kernel holds far more live than the result. Trusting the naive
    #: figure is what let a 16 GiB budget admit N=1024, whose true peak is near
    #: 70 GiB; the worker was OOM-killed before writing anything.
    #:
    #:   N=256   6.62 GiB measured   (naive 0.39, 17.0x)
    #:   N=512  19.18 GiB measured   (naive 1.56, 12.3x)
    #:
    #: Fitting peak = FIXED + SLOPE * n_segments * N^2 * 8 through those two
    #: points, with the slope taken at the lower (less flattering) ratio:
    APERTURE_FIXED_BYTES = int(2.4 * 2**30)     # tracing/compiling 798 apertures
    APERTURE_STACK_MULTIPLIER = 12.0            # live buffers per naive stack byte

    @classmethod
    def _aperture_peak_bytes(cls, case: Case) -> float:
        naive = float(case.segmented.n_segments) * case.n_pupil * case.n_pupil * 8.0
        return cls.APERTURE_FIXED_BYTES + cls.APERTURE_STACK_MULTIPLIER * naive

    def versions(self) -> dict[str, str]:
        import jax
        out = {"jax": jax.__version__, "numpy": np.__version__}
        try:
            import dLux
            out["dLux"] = getattr(dLux, "__version__", "unknown")
        except ImportError:
            out["dLux"] = "not installed"
        return out

    #: JAX differentiates the whole dLux optical system, so the forward model
    #: returns (loss, dloss/dtheta) and the optimiser never forms a difference
    #: quotient.
    retrieval_gradient = "analytic"

    #: The optimisation runs inside jax.lax.while_loop on whatever device the
    #: config selects, so unlike the scipy-driven adapters nothing crosses the
    #: host boundary per iteration -- only the final theta comes back. That is a
    #: real architectural difference on the GPU board and not a measurement
    #: artefact: it is the reason to write a retrieval in JAX.
    retrieval_devices = ("cpu", "gpu")

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.is_retrieval:
            sup = self.retrieval_support(case, config)
            if not sup:
                return sup
            try:
                import optax  # noqa: F401
            except ImportError:
                return Unsupported(
                    "optax is not installed; the analytic board drives dLux with "
                    "optax.lbfgs rather than scipy, so that JAX never has to leave "
                    "the device between iterations")
            return True
        if case.is_aperture:
            from dragrace.apertures import APERTURE_MEMORY_BUDGET_BYTES
            need = self._aperture_peak_bytes(case)
            if need > APERTURE_MEMORY_BUDGET_BYTES:
                return Unsupported(
                    f"dLux evaluates each of the {case.segmented.n_segments} "
                    f"sub-apertures over the whole coordinate grid and stacks them "
                    f"before reducing (CompositeAperture.transmission), so the "
                    f"intermediate is n_segments x N^2 x 8 bytes = "
                    f"{need / 2**30:.1f} GiB at N={case.n_pupil} (measured model, "
                    f"not the naive stack figure -- that understates it 12-17x), "
                    f"against a {APERTURE_MEMORY_BUDGET_BYTES / 2**30:.0f} GiB budget. "
                    f"Its apertures are dynamically generated so they stay "
                    f"differentiable -- the right trade for the gradient board and "
                    f"the wrong one for drawing a pupil once."
                )
            return True
        if case.algorithm_class == "angular_spectrum":
            return Unsupported(
                "dLux's ASMPropagator applies the PARAXIAL Fresnel transfer function "
                "despite the name: it matches an internal_fresnel_tf reference to "
                "2.78e-13 and an exact angular-spectrum one only to 4.07e-6, which is "
                "precisely the exact-vs-paraxial difference. Use fresnel_d50_z1m.")
        if case.algorithm_class == "fft":
            # Declined even though dLux CAN do this, because this adapter does
            # not. AngularOpticalSystem.propagate_mono ends in `to_focus`, which
            # calls Wavefront.propagate -- dLux's own docstring for it opens
            # "Legacy MFT propagation function". So a run here would compute a
            # matrix DFT and file it on the FFT board.
            #
            # Nothing would catch that. fft_array_scan gates against
            # `internal_mft`, the same reference an MFT case uses, because both
            # must land on the same focal grid -- so the wrong algorithm returns
            # the right answer and passes. The row would then be read as "dLux's
            # FFT is faster than PROPER's", when the two are not the same
            # computation at all: reaching q=4 costs an FFT a (q*N_p)^2
            # transform and costs an MFT nothing beyond N_p^2 x N_f.
            #
            # dLux exposes Wavefront.propagate_FFT, so an honest FFT path is
            # implementable; until one exists and is verified, declining is the
            # only answer that does not publish a mislabel.
            return Unsupported(
                "this adapter reaches the focal plane through "
                "AngularOpticalSystem, whose to_focus() calls Wavefront.propagate "
                "-- documented by dLux as 'Legacy MFT propagation function'. It "
                "would compute a matrix DFT and pass this case's internal_mft "
                "gate, putting an MFT row on the FFT board. dLux does expose "
                "Wavefront.propagate_FFT; wiring it up is the fix, not relaxing "
                "this. Use mft_array_scan / mft_n1024_q4 meanwhile.")
        if case.algorithm_class not in ("matrix_dft", "fresnel_tf"):
            return Unsupported(f"no dLux path for {case.algorithm_class}")
        try:
            import jax  # noqa: F401
        except ImportError:
            return Unsupported("jax not installed")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import jax

        want64 = config.precision_override != "complex64"
        if jax.config.jax_enable_x64 != want64:
            return Unsupported(
                f"JAX_ENABLE_X64 is {jax.config.jax_enable_x64}, config needs {want64}. "
                "The flag must be set before the first jax import, so it cannot be "
                "changed here. Config.jax_env() emits it from the config's precision, "
                "and the worker applies that before importing anything -- seeing this "
                "means jax was imported earlier than the worker's preamble, or the "
                "config's own `env:` block overrode it."
            )
        want_gpu = config.is_gpu
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
        if want_gpu and not has_gpu:
            return Unsupported(f"config wants {config.device} but jax.devices()={jax.devices()}")
        self._device = jax.devices("gpu" if want_gpu else "cpu")[0]
        self._gpu = want_gpu
        # blas never applies -- XLA emits its own kernels. threads DOES apply
        # now, enforced by process affinity rather than by anything XLA exposes
        # (see the class attribute above). The FFT axis applies on a config that
        # asks for XLA -- there the config and the adapter agree, and saying
        # otherwise would put a spurious "mixed backends" caveat on a figure
        # where every line runs the same engine.
        axes = ["blas"]
        if config.fft_backend not in ("xla", "native"):
            axes.insert(0, "fft")
        self.config_axes_not_selectable = tuple(axes)
        return True

    def resolve_backend(self) -> dict:
        import jax
        return {
            "array_module": "jax.numpy",
            "fft_backend": "xla",          # XLA supplies its own; mkl_fft is not applicable
            "device": self._device.platform,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
        }

    # ------------------------------------------------------------ lifecycle --
    def _optics(self, case: Case):
        """The AngularOpticalSystem a dLux user builds and then reuses.

        dLux's documented entry point is an optical system object propagated
        with propagate_mono(), not a hand-written jnp kernel. An earlier version
        of this adapter wrote the MFT directly against jax.numpy, which measured
        XLA rather than dLux.

        The pupil goes in as a dLux.Optic carrying the harness's transmission
        and OPD arrays rather than a dLux.CircularAperture, for the same reason
        as everywhere else: the case pins a hard-edged mask and each library
        antialiases differently (docs/conventions.md).
        """
        import dLux
        import jax.numpy as jnp

        # psf_pixel_scale is angular, in arcsec: one case sample is
        # (lambda/D)/q radians. diameter is the *array* extent, not the
        # aperture's. oversample=1 keeps the computed grid the requested grid.
        pixel_scale = np.degrees(
            case.wavelength_m / case.pupil.diameter_m / case.q) * 3600.0
        return dLux.AngularOpticalSystem(
            wf_npixels=case.n_pupil,
            diameter=case.pupil.diameter_m * case.n_pupil / case.n_across,
            layers=[("pupil", dLux.Optic(
                transmission=jnp.asarray(circular_aperture(case, self.grid_centering)),
                opd=jnp.asarray(opd_waves(case, self.grid_centering) * case.wavelength_m)))],
            psf_npixels=case.n_focus,
            psf_pixel_scale=pixel_scale,
            oversample=1,
        )

    def _free_space_optics(self, case: Case):
        """LayeredOpticalSystem + ASMPropagator, dLux's plane-to-plane path.

        Pixel-centred here, unlike the focal board: AngularOpticalSystem places
        its PSF between pixels, but a LayeredOpticalSystem propagated by
        ASMPropagator keeps the illumination grid, which has a sample on axis.
        Verified both ways -- 2.78e-13 against a pixel-centred paraxial
        reference.
        """
        import dLux
        import jax.numpy as jnp

        return dLux.LayeredOpticalSystem(
            wf_npixels=case.n_pupil,
            diameter=case.dx_pupil_m * case.n_pupil,
            layers=[dLux.Optic(
                        transmission=jnp.asarray(circular_aperture(case, "pixel")),
                        opd=jnp.asarray(opd_waves(case, "pixel") * case.wavelength_m)),
                    dLux.ASMPropagator(distance=case.propagation.distance_m,
                                       spec=dLux.CoordSpec(case.n_pupil, None, None))],
        )

    def _build_aperture(self, case: Case, config: Config):
        """MultiAperture of RegPolyAperture segments, times a Spider.

        dLux has no ELT and no segmented-aperture helper, so the pupil is
        composed from its aperture layers the way its documentation teaches:
        MultiAperture combines sub-apertures additively (the "mask with multiple
        holes" case), CompoundAperture combines multiplicatively, so the segments
        go in the former and the spider multiplies the result.

        Segment centres come straight from the canonical layout rather than from
        a lattice dLux enumerates, because dLux has no segment index to match
        against -- each sub-aperture is placed by an explicit CoordTransform. So
        unlike the other adapters there is nothing here to reconcile, and the
        selection is exact by construction.

        Rotation is 0, not 30 degrees: dLux's soft_reg_polygon already puts the
        hexagon flats where the ELT wants them. Measured both ways -- IoU 0.965
        at rotation 0 against 0.905 at 30.

        Two consequences of dLux's design show up in the output and are left
        alone rather than papered over. The transmission is not bounded by 1
        (measured max 1.42): MultiAperture *adds*, and the ELT's gaps are
        sub-pixel at every size on this scan, so adjacent soft edges overlap and
        sum. And every aperture is soft-edged by construction, because that is
        what keeps them differentiable. The geometry gate binarises, so neither
        changes the verdict, and accuracy.edge_relative_l2 records the softness.
        """
        import numpy as np
        import jax
        import dLux
        import dLux.utils as dlu
        from dragrace.apertures import elt_segment_centres

        seg = case.segmented
        centres = elt_segment_centres(case.segmented_spec())
        rmax = seg.segment_vertex_to_vertex_m / 2.0        # centre-to-vertex

        segments = dLux.MultiAperture([
            dLux.RegPolyAperture(
                6, rmax,
                transformation=dLux.CoordTransform(
                    translation=(float(x), float(y)), rotation=0.0))
            for x, y in centres
        ])
        layers = [segments]
        if seg.spider_count:
            # +90 rather than the case's own +30: dLux.Spider documents its
            # angles in degrees, but measures each arm from the +y axis while
            # the case (and every other adapter here) measures from +x. Passing
            # the case's angles unchanged puts the vanes 30 degrees off, which
            # removes almost exactly the right *amount* of light in almost
            # exactly the wrong *places* -- the fill fraction still lands within
            # 0.3% of the reference, so nothing looks wrong until the geometry
            # gate reports IoU 0.929. Measured across the offsets: only +30 on
            # top of the case's 30 registers dLux's vanes onto the reference's.
            layers.append(dLux.Spider(
                seg.spider_width_m,
                np.array([seg.spider_angle_offset_deg + 90.0 + 60.0 * i
                          for i in range(seg.spider_count)])))
        pupil = dLux.CompoundAperture(layers) if len(layers) > 1 else segments

        coords = dlu.pixel_coords(case.n_pupil, case.pupil.diameter_m)
        pixel_scale = case.pupil.diameter_m / case.n_pupil
        # AOT-compiled for the same reason as the propagation boards: compile
        # time becomes setup.first_call_s instead of hiding inside a warm-up.
        compiled = jax.jit(
            lambda: pupil.transmission(coords, pixel_scale)).lower().compile()
        return {"case": case, "aperture": True, "fn": compiled}

    def _build_retrieval(self, case: Case, config: Config):
        """Untimed: the optical system, the loss, and the AOT-compiled retrieval.

        THE WHOLE OPTIMISATION IS ONE COMPILED XLA PROGRAM. That is the point of
        putting dLux on this board rather than driving it from scipy: with
        scipy's L-BFGS-B the loop lives in Python and every iteration crosses the
        host/device boundary twice, which at these array sizes costs more than
        the propagation. Compiling the loop keeps JAX on-device from the
        starting guess to convergence, and it is what a dLux user would write.
        The price is a real compile, which lands in setup.first_call_s exactly as
        it does on the propagation boards -- seconds, where prysm's is
        milliseconds. Neither number is hidden and they should be read together.

        THE OPTIMISER IS optax.lbfgs, NOT L-BFGS-B, and that is a genuine
        deviation worth naming rather than glossing. optax ships L-BFGS with a
        zoom line search and has no box-constrained variant; scipy's L-BFGS-B
        with no bounds passed reduces to the same algorithm, because the only
        thing the -B adds is a projection onto bounds that do not exist here.
        Nothing on this board is bounded, so the two solve the same problem --
        but they are different implementations with different line searches, and
        a reader comparing prysm's row against dLux's is comparing those too.

        The three convergence tests are scipy's own, transcribed rather than
        approximated, so "time to converge" means the same thing on both boards:

            gtol   max |g_i| <= gtol
            ftol   (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol
            cap    maxiter

        Without this the comparison would be silent nonsense -- optax's default
        stopping is nothing at all, so dLux would run the full iteration cap
        while the scipy-driven codes stopped at 22, and dLux would look slow for
        having done four times the work.
        """
        import jax
        import jax.numpy as jnp
        import optax
        import optax.tree_utils as otu
        import dLux

        from dragrace.grid import aperture_mask
        from dragrace.retrieval import loss_scale, retrieval_parameters

        r = case.retrieval
        _, theta_true, theta_init, basis = retrieval_parameters(
            case, self.grid_centering)
        basis_d = jnp.asarray(basis)
        amp = jnp.asarray(aperture_mask(case, self.grid_centering))
        wavelength = case.wavelength_m

        # psf_pixel_scale is angular, in arcsec: one case sample is
        # (lambda/D)/q radians. diameter is the ARRAY extent, not the aperture's.
        pixel_scale = np.degrees(
            case.wavelength_m / case.pupil.diameter_m / case.q) * 3600.0
        optics = dLux.AngularOpticalSystem(
            wf_npixels=case.n_pupil,
            diameter=case.pupil.diameter_m * case.n_pupil / case.n_across,
            layers=[("pupil", dLux.Optic(transmission=amp,
                                         opd=jnp.zeros_like(amp)))],
            psf_npixels=case.n_focus,
            psf_pixel_scale=pixel_scale,
            oversample=1,
        )

        def psf(theta):
            # zodiax's .set returns a new pytree with one leaf replaced, which
            # is how a dLux model is updated inside a differentiated function --
            # rebuilding the OpticalSystem per call would trace its constructor
            # instead.
            opd = jnp.tensordot(theta, basis_d, axes=(0, 0)) * wavelength
            return optics.set("pupil.opd", opd).propagate_mono(wavelength)

        observed = jax.block_until_ready(psf(jnp.asarray(theta_true)))
        s = loss_scale(np.asarray(observed))

        def loss(theta):
            return jnp.mean(((psf(theta) - observed) / s) ** 2)

        opt = optax.lbfgs(memory_size=r.history_size)
        value_and_grad = optax.value_and_grad_from_state(loss)
        ftol, gtol, maxiter = r.ftol, r.gtol, r.max_iterations

        def retrieve(theta0):
            def cond(carry):
                _, _, _, i, _, stop = carry
                return jnp.logical_and(i < maxiter, jnp.logical_not(stop))

            def body(carry):
                params, st, prev, i, n_ls, _ = carry
                value, grad = value_and_grad(params, state=st)
                denom = jnp.maximum(
                    jnp.maximum(jnp.abs(prev), jnp.abs(value)), 1.0)
                stop = jnp.logical_or(
                    jnp.max(jnp.abs(grad)) <= gtol,
                    jnp.logical_and(i > 0, (prev - value) / denom <= ftol))
                updates, st2 = opt.update(grad, st, params, value=value,
                                          grad=grad, value_fn=loss)
                stepped = optax.apply_updates(params, updates)
                # Hold the point rather than take one more step once converged,
                # so the returned coefficients are the ones the stopping test
                # was actually satisfied at.
                params = jnp.where(stop, params, stepped)
                n_ls = n_ls + otu.tree_get(st2, "num_linesearch_steps")
                return (params, st2, value, i + 1, n_ls, stop)

            init = (theta0, opt.init(theta0), jnp.asarray(jnp.inf),
                    jnp.asarray(0), jnp.asarray(0, dtype=jnp.int64),
                    jnp.asarray(False))
            params, _, _, n_iter, n_ls, stop = jax.lax.while_loop(cond, body, init)
            return params, loss(params), n_iter, n_ls, stop

        theta0 = jax.device_put(jnp.asarray(theta_init), self._device)
        compiled = jax.jit(retrieve).lower(theta0).compile()

        state = {"case": case, "retrieval": True, "fn": compiled, "theta0": theta0,
                 "psf": lambda th: np.asarray(psf(jnp.asarray(th, dtype=float))),
                 # The traceable loss, kept on the state so the gradient can be
                 # checked against central differences directly rather than
                 # inferred from whether the retrieval converged. A wrong
                 # gradient still converges, only slower -- which on a timing
                 # board is the one failure that must not go unnoticed.
                 "loss": loss,
                 "loss_initial": float(loss(theta0))}
        try:
            state["cost_analysis"] = compiled.cost_analysis()
        except Exception as exc:                     # noqa: BLE001
            state["cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def retrieval_psf(self, state, theta) -> np.ndarray:
        return state["psf"](theta)

    def retrieval_report(self, state, result) -> dict:
        from dragrace.retrieval import make_report
        return make_report(
            result, state["loss_initial"], state["case"],
            state["case"].n_focus ** 2,
            forward_model="dLux AngularOpticalSystem under jax.value_and_grad",
            n_fev_note=(
                "derived from optax's own num_linesearch_steps counter rather "
                "than measured directly: the loop is one compiled XLA program, so "
                "there is no host-side call to count. One initial evaluation, one "
                "per line-search step, and one final loss. "
                "optax.value_and_grad_from_state reuses the value the line search "
                "already computed, so iterations themselves add none."))

    def build(self, case: Case, config: Config):
        import jax

        if case.is_retrieval:
            return self._build_retrieval(case, config)

        if case.is_aperture:
            return self._build_aperture(case, config)

        if case.kind == "plane_to_plane":
            self.grid_centering = "pixel"
            optics = self._free_space_optics(case)
            wavelength = jax.device_put(case.wavelength_m, self._device)
            compiled = jax.jit(
                lambda wl: optics.propagate_mono(wl)).lower(wavelength).compile()
            return {"case": case, "optics": optics, "wavelength": wavelength,
                    "fn": compiled}

        optics = self._optics(case)
        wavelength = jax.device_put(case.wavelength_m, self._device)

        # jit is not a deviation from the documented usage: dLux's own examples
        # jit their propagations, and an unjitted call would measure equinox
        # pytree traversal per invocation rather than the propagation. Lowering
        # and compiling here keeps compile time a first-class number
        # (setup.first_call_s) instead of hiding it in a warm-up.
        lowered = jax.jit(lambda wl: optics.propagate_mono(wl)).lower(wavelength)
        compiled = lowered.compile()

        state = {"case": case, "optics": optics, "wavelength": wavelength,
                 "fn": compiled}
        try:
            state["cost_analysis"] = compiled.cost_analysis()
            state["memory_analysis"] = str(compiled.memory_analysis())
        except Exception as exc:                     # noqa: BLE001
            state["cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def propagate(self, state):
        """propagate_mono returns the PSF (intensity), as dLux documents."""
        if state.get("retrieval"):
            from dragrace.retrieval import Outcome

            params, loss, n_iter, n_ls, stop = state["fn"](state["theta0"])
            n_ls = int(n_ls)
            return Outcome(
                theta=np.asarray(params), loss_initial=float("nan"),
                loss_final=float(loss), n_iterations=int(n_iter),
                # One initial evaluation, one per line-search step, one final
                # loss. See retrieval_report for why this is derived rather
                # than counted.
                n_fev=n_ls + 2, n_jev=n_ls + 1, converged=bool(stop),
                message=("converged on the case's ftol/gtol tests"
                         if bool(stop) else
                         "hit retrieval.max_iterations without meeting ftol or gtol"),
                optimizer="optax.lbfgs + zoom line search, in one compiled XLA program",
            )
        if state.get("aperture"):
            return state["fn"]()
        return state["fn"](state["wavelength"])

    def complex_field(self, state, result) -> np.ndarray:
        """Untimed: dLux's documented `return_wf=True` hands back the Wavefront.

        An aperture case has already produced the gated quantity -- a drawn
        transmission mask -- so there is no field to recover.

        Same propagation, but it returns the object rather than the intensity,
        so it is kept out of the clock exactly as POPPY's return_final is.
        """
        import jax.numpy as jnp

        if state.get("aperture"):
            return self.to_host(result)
        wf = state["optics"].propagate_mono(state["wavelength"], return_wf=True)
        phasor = getattr(wf, "phasor", None)
        if phasor is None:
            phasor = wf.amplitude * jnp.exp(1j * wf.phase)
        return np.asarray(phasor).astype(state["case"].dtype)

    def sync(self, result) -> None:
        import jax
        # Must block on every leaf: a pytree return would otherwise leave part
        # of the computation outstanding when the clock stops. A retrieval has
        # already been materialised into an Outcome of host values by
        # propagate() -- reading n_iterations out of the device array is itself
        # a blocking transfer, and it happens inside the clock.
        if not hasattr(result, "theta"):
            jax.block_until_ready(result)

    def to_host(self, result) -> np.ndarray:
        if hasattr(result, "theta"):
            return np.asarray(result.theta)
        return np.asarray(result)

    def device_memory(self):
        try:
            import jax
            stats = self._device.memory_stats() or {}
            return int(stats.get("peak_bytes_in_use", 0)) or None
        except Exception:                            # noqa: BLE001
            return None

    # ------------------------------------------------------ gradient board --
    def supports_gradient(self) -> bool | Unsupported:
        try:
            import jax  # noqa: F401
        except ImportError:
            return Unsupported("jax not installed")
        return True

    def build_gradient(self, case: Case, config: Config):
        import jax
        import jax.numpy as jnp

        noll, theta0, basis = gradient_parameters(case)
        basis_d = jnp.asarray(basis)
        amp = jnp.asarray(circular_aperture(case))
        x = jnp.asarray(pupil_coords(case))
        u = jnp.asarray(focus_coords(case))
        scale = case.dx_pupil ** 2

        def forward_intensity(theta):
            phs = jnp.tensordot(theta, basis_d, axes=(0, 0))
            w = amp * jnp.exp(2j * jnp.pi * phs)
            kx = jnp.exp(-2j * jnp.pi * jnp.outer(u, x)).astype(w.dtype)
            e = (kx @ w) @ kx.T * scale
            return jnp.abs(e) ** 2

        target = jax.block_until_ready(forward_intensity(jnp.zeros_like(jnp.asarray(theta0))))

        def loss(theta):
            return jnp.mean((forward_intensity(theta) - target) ** 2)

        theta_d = jax.device_put(jnp.asarray(theta0), self._device)
        lowered = jax.jit(jax.value_and_grad(loss)).lower(theta_d)
        compiled = lowered.compile()

        state = {"case": case, "theta": np.asarray(theta0), "theta_d": theta_d,
                 "fn": compiled}
        try:
            # The board's headline number: XLA's own count for the gradient,
            # comparable against prysm's hand-written adjoint on the same case.
            state["cost_analysis_grad"] = compiled.cost_analysis()
            fwd = jax.jit(loss).lower(theta_d).compile()
            state["cost_analysis_fwd"] = fwd.cost_analysis()
        except Exception as exc:                     # noqa: BLE001
            state["cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def gradient(self, state):
        loss, grad = state["fn"](state["theta_d"])
        return float(loss), np.asarray(grad)

    def gradient_theta(self, state) -> np.ndarray:
        return state["theta"]
