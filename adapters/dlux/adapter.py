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
from dragrace.grid import circular_aperture, focus_coords, gradient_parameters, pupil_coords


@register("dlux")
class DLuxAdapter(Adapter):
    status = "unverified"
    reviewed_by = ""             # invite Louis Desdoigts before publishing results
    requires = ("jax", "dLux")

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
        if case.algorithm_class not in ("matrix_dft", "fft"):
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
                "This flag must be set before the first jax import, so it cannot be "
                "changed here -- set it in the environment (scripts/setup_env.sh writes "
                "it into activate.d) and re-run."
            )
        want_gpu = config.is_gpu
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
        if want_gpu and not has_gpu:
            return Unsupported(f"config wants {config.device} but jax.devices()={jax.devices()}")
        self._device = jax.devices("gpu" if want_gpu else "cpu")[0]
        self._gpu = want_gpu
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
    def _forward_fn(self, case: Case):
        """Pupil field -> focal field, as a pure JAX function.

        Written against jax.numpy directly rather than through dLux's model
        objects for the forward board, so that what is being timed is the
        propagation rather than dLux's OpticalSystem construction. A dLux-native
        variant belongs alongside this one -- it would measure a different and
        also interesting thing (the cost of the framework's abstractions), and
        should be a separate adapter rather than silently swapped in here.
        """
        import jax.numpy as jnp

        x = jnp.asarray(pupil_coords(case))
        u = jnp.asarray(focus_coords(case))
        scale = case.dx_pupil ** 2

        def fwd(field):
            kx = jnp.exp(-2j * jnp.pi * jnp.outer(u, x)).astype(field.dtype)
            return (kx @ field) @ kx.T * scale

        return fwd

    def build(self, case: Case, config: Config):
        import jax
        import jax.numpy as jnp
        from dragrace.grid import pupil_field

        field = jnp.asarray(pupil_field(case))
        field = jax.device_put(field, self._device)

        fn = self._forward_fn(case)
        lowered = jax.jit(fn).lower(field)
        compiled = lowered.compile()                 # compile happens here, untimed

        state = {"case": case, "field": field, "fn": compiled}
        try:
            state["cost_analysis"] = compiled.cost_analysis()
            state["memory_analysis"] = str(compiled.memory_analysis())
        except Exception as exc:                     # noqa: BLE001
            state["cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def propagate(self, state):
        return state["fn"](state["field"])

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
