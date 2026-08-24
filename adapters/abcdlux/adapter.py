"""abcdLux adapter (JAX/XLA) -- the same author's functional propagator backend.

WHY THIS SITS BESIDE THE dLux ADAPTER RATHER THAN REPLACING IT. abcdLux
(github.com/LouisDesdoigts/abcdLux) is not a newer dLux. It is a library of
*functions* -- coordinates, ABCD matrices, MFT kernels, Fraunhofer/ASM/LCT
propagators -- with no optical-system object and no pytree model. dLux is the
model layer: an AngularOpticalSystem you build, differentiate, and hand to an
optimiser. Putting both on the board therefore measures one specific design
difference and nothing else, because they share an author, a language and an
execution engine:

    dLux      the kernel is rebuilt inside every propagation.
    abcdLux   the kernel is a value the caller holds.

THAT SPLIT IS abcdLux's OWN API, NOT THIS ADAPTER TAKING A LIBERTY. Every
propagator in the library ships as a pair:

    S, Kx, Ky = fraunhofer_kernels(spec_in, spec_out, lam, f)   # build once
    u_f       = fraunhofer_kernel_prop(u_p, S, Kx, Ky)          # call many times

    fraunhofer_prop(u_p, spec_in, spec_out, lam, f)             # both, per call

and the harness's build()/propagate() split lands exactly on the seam the
library already cut. `fraunhofer_prop` is the convenience wrapper; using it here
would time the kernel construction on every repeat and measure the wrapper
rather than the propagator. See docs/methodology.md on what belongs in build():
"anything a real user would hoist out of a loop". abcdLux gives that phrase a
function signature.

WHAT dLux DOES INSTEAD, AND WHY IT IS NOT A DEFECT. dLux.utils.propagation.MFT
calls `vmap(get_tf_mat)(shift)` on every invocation, so both (N_f, N_p) transfer
matrices are re-exponentiated inside the timed region -- 2*N_f*N_p complex
exponentials per propagation, growing linearly in N_p while the two GEMMs that
follow grow quadratically. It cannot be hoisted without changing what dLux is:
the matrices depend on `wavelength`, which is the traced input of
`propagate_mono`, and a dLux user differentiates and vmaps over wavelength. The
cost buys polychromatic sources and wavelength-differentiable models. abcdLux
buys neither and charges neither. The two rows on the scan figure are that
trade, priced.

CENTRING. Interpixel, and measured rather than assumed: abcdLux.coords.nd_coords
returns linspace(-(N-1)/2*d, +(N-1)/2*d, N), which for even N puts no sample on
the axis. Both planes use it, so unlike HCIPy the two agree. Against the
harness's references at mft_n1024_q4:

    interpixel   rel_l2 = 2.36e-15
    pixel        rel_l2 = 2.82e-01

NORMALISATION. abcdLux carries a throughput-preserving scale,
S = sqrt(dx_in*dx_out*dy_in*dy_out)/(lam*f), where this suite's reference
carries dx^2 (dragrace.reference.reference_mft). The two differ by exactly
n_across/q -- measured 256.000 at n_across=1024, q=4. That is a convention, not
an error, and the accuracy gate fits one complex scale before computing rel_l2
precisely so that a disagreement about where the 1/(lambda f) lives cannot be
mistaken for a disagreement about the optics. It is reported as `scale_abs`.

X64. As for dLux: JAX defaults to complex64 and JAX_ENABLE_X64 must be set
before the first jax import. Config.jax_env() emits it and the worker applies it
before importing anything; configure() refuses rather than silently downcasting.

THREADS. XLA honours no thread environment variable. The axis is enforced by
dragrace.worker._pin_cpus through CPU affinity, which is a property of the
process that XLA cannot opt out of -- see the long note in the dLux adapter for
what happened before that was true.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import pupil_field


@register("abcdlux")
class ABCDLuxAdapter(Adapter):
    #: Exercised against abcdLux 0.0.2 on this machine: mft_n1024_q4 and
    #: mft_array_scan at every point, and fresnel_d50_z1m / fresnel_scan_z1m,
    #: all gated against the harness's own references.
    status = "verified"
    #: abcdLux and dLux share an author. Neither adapter has been reviewed by
    #: him yet, and both rows should carry that until they have -- an adapter
    #: author is not a neutral party (docs/methodology.md).
    reviewed_by = ""             # invite Louis Desdoigts before publishing results
    requires = ("jax", "abcdLux")

    #: Measured, not assumed. See the module docstring.
    grid_centering = "interpixel"

    #: XLA emits its own kernels: there is no BLAS to point at and no FFT
    #: library to select, so a config's `blas` axis never applies and its `fft`
    #: axis applies only when it asks for XLA. Same declaration as the dLux
    #: adapter, for the same reason and with the same consequence: an abcdLux
    #: row is legitimate on cpu_numpy_1t but is never a data point in a *backend*
    #: comparison. `threads` is absent because affinity makes it true.
    config_axes_not_selectable = ("fft", "blas")   # refined in configure()

    def versions(self) -> dict[str, str]:
        import jax
        out = {"jax": jax.__version__, "numpy": np.__version__}
        try:
            import abcdLux
            out["abcdLux"] = getattr(abcdLux, "__version__", "unknown")
        except ImportError:
            out["abcdLux"] = "not installed"
        # Recorded even though this adapter never imports dLux, because the
        # figure it exists for is a two-row comparison and the reader needs both
        # versions on the same result. abcdLux vendors its own nd_coords and has
        # no dLux dependency -- `pip show abcdLux` lists jax and jaxlib only.
        try:
            import dLux
            out["dLux_present"] = getattr(dLux, "__version__", "unknown")
        except ImportError:
            out["dLux_present"] = "not installed"
        return out

    # ------------------------------------------------------------- support --
    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.is_retrieval:
            # retrieval_gradient is None, so the base class produces the
            # standard refusal. Stated plainly rather than left implicit: this
            # is a statement about the adapter. abcdLux is JAX end to end and
            # jax.value_and_grad differentiates fraunhofer_kernel_prop without
            # modification, so a retrieval path is implementable -- it is not
            # written, and an unwritten path must not be advertised.
            return self.retrieval_support(case, config)
        if case.is_aperture:
            return Unsupported(
                "abcdLux is a propagator library: it has coordinates, ABCD matrices "
                "and Fourier/Fresnel kernels, and no aperture layer at all. The "
                "aperture board times drawing a pupil, which is work abcdLux does not "
                "claim to do. Use the dlux adapter for that comparison.")
        if case.algorithm_class == "fft":
            # abcdLux does expose an FFT propagator (lct.lct_prop_fft), and it
            # would be a fair entry on this board. It is not wired up here, and
            # the reason for declining rather than routing the case through the
            # MFT path is the same one the dLux adapter gives at greater length:
            # fft_array_scan gates against internal_mft, so an MFT answer would
            # pass the gate and file itself on the FFT board, where it would be
            # read as "abcdLux's FFT is faster than PROPER's" while not being an
            # FFT at all.
            return Unsupported(
                "not wired up. abcdLux has lct.lct_prop_fft, but this adapter reaches "
                "the focal plane only through fraunhofer_kernel_prop, which is a "
                "matrix DFT. Routing an FFT case through it would pass this case's "
                "internal_mft gate and put an MFT row on the FFT board. Wiring up "
                "lct_prop_fft is the fix; use mft_array_scan / mft_n1024_q4 meanwhile.")
        if case.algorithm_class == "angular_spectrum":
            # abcdLux's own docstring for asm_kernels is explicit: "the Fresnel
            # angular-spectrum transfer function", H = exp(-i pi lam z (fx^2+fy^2)).
            # That is the paraxial kernel under an ASM name -- the identical
            # situation as dLux's ASMPropagator, and the reference this suite
            # would gate it against is the exact one.
            return Unsupported(
                "abcdLux.asm_kernels builds the PARAXIAL kernel despite the module "
                "name -- its own docstring gives H = exp(-i pi lam z (fx^2 + fy^2)), "
                "which is fresnel_tf, not the exact sqrt(1 - (lam f)^2) form. Measured "
                "against the paraxial reference it scores 1.6e-15; against the exact "
                "one it would show the 4.07e-6 paraxial-vs-exact difference and be "
                "marked wrong for a choice the library is open about. Use "
                "fresnel_d50_z1m / fresnel_scan_z1m.")
        if case.algorithm_class not in ("matrix_dft", "fresnel_tf"):
            return Unsupported(f"no abcdLux path for {case.algorithm_class}")
        try:
            import jax          # noqa: F401
            import abcdLux      # noqa: F401
        except ImportError as exc:
            return Unsupported(f"{exc}")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        import jax

        want64 = config.precision_override != "complex64"
        if jax.config.jax_enable_x64 != want64:
            return Unsupported(
                f"JAX_ENABLE_X64 is {jax.config.jax_enable_x64}, config needs {want64}. "
                "The flag must be set before the first jax import, so it cannot be "
                "changed here. Config.jax_env() emits it from the config's precision "
                "and the worker applies that before importing anything -- seeing this "
                "means jax was imported earlier than the worker's preamble, or the "
                "config's own `env:` block overrode it."
            )
        want_gpu = config.is_gpu
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
        if want_gpu and not has_gpu:
            return Unsupported(
                f"config wants {config.device} but jax.devices()={jax.devices()}")
        self._device = jax.devices("gpu" if want_gpu else "cpu")[0]
        axes = ["blas"]
        if config.fft_backend not in ("xla", "native"):
            axes.insert(0, "fft")
        self.config_axes_not_selectable = tuple(axes)
        return True

    def resolve_backend(self) -> dict:
        import jax
        return {
            "array_module": "jax.numpy",
            "fft_backend": "xla",
            "device": self._device.platform,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
        }

    # ------------------------------------------------------------ lifecycle --
    def build(self, case: Case, config: Config):
        """Untimed: the pupil field, the kernels, and the compiled propagation.

        THE PUPIL FIELD IS HOISTED, as it is for every NumPy-backed adapter in
        this suite (dragrace.grid.pupil_field, called in build()). These cases
        benchmark propagation; rasterisation and the phasor are charged to the
        gradient and aperture boards instead. abcdLux can take that hoist because
        it propagates a *field* -- the dLux adapter cannot, because
        propagate_mono takes a wavelength and reconstructs the phasor from
        transmission and OPD inside the traced function. That is a second real
        difference between the two rows and is named here so it is not read off
        the figure as part of the first one.

        AOT COMPILATION, matching the dLux adapter exactly: lower().compile() so
        the timed region is pure execution and the compile lands in
        setup.first_call_s as a first-class number instead of hiding inside a
        warm-up. Comparing a compiled dLux against an eager abcdLux would
        measure equinox pytree traversal against XLA dispatch and call it a
        propagator difference.

        THE FIELD IS AN ARGUMENT AND THE KERNELS ARE CLOSED OVER. That is what
        an abcdLux user writes, and it is also the only arrangement that
        measures anything: with the field closed over too, the jitted function
        would take no arguments and XLA would constant-fold the entire
        propagation at compile time, leaving a timed region that returns a
        literal. One argument, matching dLux's one argument, so the dispatch
        cost is the same on both rows.
        """
        import abcdLux
        import jax
        import jax.numpy as jnp

        field = jax.device_put(
            jnp.asarray(pupil_field(case, centering=self.grid_centering)), self._device)

        if case.algorithm_class == "fresnel_tf":
            # asm_kernels returns the kernel on the *unshifted* FFT grid, and
            # asm_kernel_prop is a plain fft2 -> multiply -> ifft2 with no
            # fftshift anywhere. That is correct rather than an omission: a
            # circular shift of the input becomes a phase ramp under fft2, the
            # ramp commutes with a multiplication by H (which is a function of
            # frequency alone), and ifft2 undoes it -- so the output comes back
            # on whatever grid the input was given on. Same reason prysm and
            # HCIPy pay for no shifts here and PROPER pays for six.
            # npad=None: this case declares its own guard band in array_samples,
            # so there is nothing to pad and crop_to is a no-op.
            H, nx, ny = abcdLux.asm_kernels(
                spec_in=(case.n_pupil, case.dx_pupil_m),
                lam=case.wavelength_m,
                z=case.propagation.distance_m,
                npad=None,
            )
            H = jax.device_put(H, self._device)
            fn = jax.jit(lambda u: abcdLux.asm_kernel_prop(u, H, nx, ny))
        else:
            # (S, Kx, Ky) -- the whole point of this adapter. Kx is (N_f, N_p)
            # and Ky is (N_f, N_p); abcdLux keeps the two axes separate rather
            # than assuming a square symmetric kernel, and applies them as
            # `Ky @ (u @ Kx.T)`. Identical FLOPs to the baseline's
            # `(kx @ w) @ kx.T`, and it is what lets a non-square or anisotropic
            # output grid cost nothing extra.
            S, Kx, Ky = abcdLux.fraunhofer_kernels(
                spec_in=(case.n_pupil, case.dx_pupil_m),
                spec_out=(case.n_focus, case.dx_focus_m),
                lam=case.wavelength_m,
                f=case.output.focal_length_m,
            )
            S = jax.device_put(S, self._device)
            Kx = jax.device_put(Kx, self._device)
            Ky = jax.device_put(Ky, self._device)
            fn = jax.jit(
                lambda u: abcdLux.fraunhofer_kernel_prop(u, S, Kx, Ky))

        compiled = fn.lower(field).compile()
        state = {"case": case, "field": field, "fn": compiled}
        try:
            # XLA's own arithmetic count, straight from the compiled program.
            # On this board it is the number that makes the dLux comparison
            # readable without a profiler: the two rows run the same two GEMMs,
            # so whatever separates their FLOP counts is the kernel rebuild.
            state["cost_analysis"] = compiled.cost_analysis()
            state["memory_analysis"] = str(compiled.memory_analysis())
        except Exception as exc:                     # noqa: BLE001
            state["cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def propagate(self, state):
        """One call to abcdLux's precomputed-kernel propagator.

        Returns the complex focal field, which is what abcdLux hands back --
        no intensity, so `complex_field` needs no override and the gate reads
        phase and normalisation directly.
        """
        return state["fn"](state["field"])

    def sync(self, result) -> None:
        import jax
        jax.block_until_ready(result)

    def to_host(self, result) -> np.ndarray:
        return np.asarray(result)

    def device_memory(self):
        try:
            stats = self._device.memory_stats() or {}
            return int(stats.get("peak_bytes_in_use", 0)) or None
        except Exception:                            # noqa: BLE001
            return None

    # ------------------------------------------------------ gradient board --
    def supports_gradient(self) -> bool | Unsupported:
        return Unsupported(
            "not written. abcdLux is JAX end to end and jax.value_and_grad "
            "differentiates fraunhofer_kernel_prop unmodified, so this board is "
            "reachable -- but the gradient case aberrates the pupil through a "
            "Zernike basis, and abcdLux has no pupil layer to differentiate "
            "through. The forward model would have to be hand-written against "
            "jax.numpy, at which point the row would measure XLA rather than "
            "abcdLux. Use the dlux adapter, whose optical system is the thing "
            "that board is asking about.")
