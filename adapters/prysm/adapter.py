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

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("matrix_dft", "fft", "czt"):
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
            else:
                mathops.fft._srcmodule = _sfft

        pconf.precision = 32 if config.precision_override == "complex64" else 64
        self._fft_name = config.fft_backend
        return True

    def resolve_backend(self) -> dict:
        from prysm import mathops
        from dragrace.backend import detect_blas
        arr = getattr(mathops.np._srcmodule, "__name__", "?")
        return {
            "array_module": arr,
            "fft_module": getattr(mathops.fft._srcmodule, "__name__", "?"),
            "fft_backend": self._fft_name,
            "device": "cuda" if "cupy" in arr else "cpu",
            "blas": "unknown" if self._gpu else detect_blas(),
        }

    # ------------------------------------------------------------ lifecycle --
    def build(self, case: Case, config: Config):
        from prysm.propagation import prepare_executor

        field = pupil_field(case)
        if self._gpu:
            import cupy as cp
            field = cp.asarray(field)

        state = {"case": case, "field": field, "cls": case.algorithm_class}
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
        from prysm.propagation import focus, focus_dft

        if state["cls"] in ("matrix_dft", "czt"):
            return focus_dft(state["field"], state["executor"])
        out = focus(state["field"], state["Q"])
        c, n = state["crop"]
        return out[c:c + n, c:c + n]

    def sync(self, result) -> None:
        if self._gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    def to_host(self, result) -> np.ndarray:
        if self._gpu:
            import cupy as cp
            return cp.asnumpy(result)
        return np.asarray(result)

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
        ibar = 2.0 * resid / resid.size
        ebar = ibar * xp.conj(e)                    # dL/dE* for I = |E|^2
        wbar = focus_dft_adjoint(ebar, executor)    # adjoint of the MDFT
        phsbar = -4.0 * xp.pi * xp.imag(wbar * w)   # W = amp*exp(2i.pi.phs)
        grad = sum_of_2d_modes_adjoint(basis, phsbar)
        return loss, xp.asarray(grad).real

    def gradient_theta(self, state) -> np.ndarray:
        return np.asarray(state["theta"])
