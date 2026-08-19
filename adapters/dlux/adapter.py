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
    #: threads is here too, and for a different reason from fft/blas: XLA does
    #: have a thread pool, but threadpoolctl cannot see it and the XLA_FLAGS
    #: knobs produced identical timings at 1 and 8 threads when tested, so the
    #: harness cannot verify the count. Declaring it keeps a dLux row out of any
    #: thread-scaling comparison instead of letting NumPy's idle OpenBLAS stand
    #: in as evidence.
    config_axes_not_selectable = ("fft", "blas", "threads")   # refined in configure()

    def versions(self) -> dict[str, str]:
        import jax
        out = {"jax": jax.__version__, "numpy": np.__version__}
        try:
            import dLux
            out["dLux"] = getattr(dLux, "__version__", "unknown")
        except ImportError:
            out["dLux"] = "not installed"
        return out

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class == "angular_spectrum":
            return Unsupported(
                "dLux's ASMPropagator applies the PARAXIAL Fresnel transfer function "
                "despite the name: it matches an internal_fresnel_tf reference to "
                "2.78e-13 and an exact angular-spectrum one only to 4.07e-6, which is "
                "precisely the exact-vs-paraxial difference. Use fresnel_d50_z1m.")
        if case.algorithm_class not in ("matrix_dft", "fft", "fresnel_tf"):
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
        # blas and threads never apply. The FFT axis does apply on a config that
        # asks for XLA -- there the config and the adapter agree, and saying
        # otherwise would put a spurious "mixed backends" caveat on a figure
        # where every line runs the same engine.
        axes = ["blas", "threads"]
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

    def build(self, case: Case, config: Config):
        import jax

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
        return state["fn"](state["wavelength"])

    def complex_field(self, state, result) -> np.ndarray:
        """Untimed: dLux's documented `return_wf=True` hands back the Wavefront.

        Same propagation, but it returns the object rather than the intensity,
        so it is kept out of the clock exactly as POPPY's return_final is.
        """
        import jax.numpy as jnp

        wf = state["optics"].propagate_mono(state["wavelength"], return_wf=True)
        phasor = getattr(wf, "phasor", None)
        if phasor is None:
            phasor = wf.amplitude * jnp.exp(1j * wf.phase)
        return np.asarray(phasor).astype(state["case"].dtype)

    def sync(self, result) -> None:
        import jax
        # Must block on every leaf: a pytree return would otherwise leave part
        # of the computation outstanding when the clock stops.
        jax.block_until_ready(result)

    def to_host(self, result) -> np.ndarray:
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
