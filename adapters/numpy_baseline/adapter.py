"""NumPy baseline: the performance floor and the harness's own test subject.

Not a competitor. This adapter exists so that (a) the whole pipeline is
runnable with no propagator installed, (b) the analytic FLOP model can be
validated against an implementation whose every operation is visible, and
(c) the other six have a "what does this cost if you just write it down"
reference to be measured against.

Deliberately naive: no plan caching beyond what NumPy does internally, no
clever memory reuse. If a real propagator cannot beat this, that is a finding.
"""
from __future__ import annotations

import numpy as np

from dragrace.adapter import Adapter, Unsupported, register
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import (
    circular_aperture,
    focus_coords,
    gradient_parameters,
    opd_waves,
    pupil_coords,
    pupil_field,
)


def _select_fft(config: Config):
    """Return (fft2_callable, resolved_name).

    Resolution is reported, never assumed -- a requested backend that is not
    installed must surface as a mismatch rather than as a silent fallback that
    mislabels the result.
    """
    want = config.fft_backend
    if want == "numpy":
        return np.fft.fft2, "numpy"
    if want == "scipy_pocketfft":
        import scipy.fft as sfft
        return sfft.fft2, "scipy_pocketfft"
    if want == "mkl":
        import mkl_fft            # noqa: F401
        from mkl_fft import _numpy_fft as mfft
        return mfft.fft2, "mkl"
    if want == "pyfftw":
        import pyfftw.interfaces.numpy_fft as pfft
        import pyfftw
        pyfftw.interfaces.cache.enable()
        return pfft.fft2, "pyfftw"
    raise ValueError(f"unsupported fft_backend {want!r}")


@register("numpy_baseline")
class NumpyBaselineAdapter(Adapter):
    status = "verified"
    reviewed_by = "harness authors (this is the reference implementation)"
    requires = ("numpy",)

    def __init__(self) -> None:
        self._fft2 = np.fft.fft2
        self._fft_name = "numpy"

    # ------------------------------------------------------------ metadata --
    def versions(self) -> dict[str, str]:
        return {"numpy": np.__version__, "adapter": "0.1.0"}

    def supports(self, case: Case, config: Config) -> bool | Unsupported:
        if case.algorithm_class not in ("matrix_dft", "fft", "angular_spectrum",
                                        "fresnel_tf"):
            return Unsupported(
                f"baseline implements matrix_dft, fft, angular_spectrum and "
                f"fresnel_tf, not {case.algorithm_class}")
        if config.is_gpu:
            return Unsupported("CPU only by construction; this is the NumPy floor")
        return True

    def configure(self, config: Config) -> bool | Unsupported:
        try:
            self._fft2, self._fft_name = _select_fft(config)
        except ImportError as exc:
            return Unsupported(f"fft_backend {config.fft_backend!r} unavailable: {exc}")
        return True

    def resolve_backend(self) -> dict:
        from dragrace.backend import detect_blas
        return {"fft_backend": self._fft_name, "array_module": "numpy",
                "device": "cpu", "blas": detect_blas()}

    # ------------------------------------------------------------ lifecycle --
    def build(self, case: Case, config: Config):
        """Untimed: pupil field, DFT kernels, padded scratch.

        The pupil field is built here rather than in propagate() because these
        cases benchmark *propagation*. Aperture rasterisation and phasor
        construction are charged to the gradient board instead, where they are
        genuinely part of the differentiated chain.
        """
        field = pupil_field(case)
        state = {"case": case, "field": field, "cls": case.algorithm_class}

        if case.algorithm_class in ("angular_spectrum", "fresnel_tf"):
            # Whichever kernel the case names -- exact or paraxial. The transfer
            # function is hoisted: it depends only on the geometry, so a user
            # propagating repeatedly builds it once. Whether a library does that
            # is exactly the kind of design difference the ledger and the
            # setup/steady split exist to expose.
            from dragrace.reference import free_space_transfer_function
            state["cls"] = "free_space"
            state["H"] = free_space_transfer_function(
                case, case.algorithm_class).astype(case.dtype)
            # Stored unshifted so propagate() is a plain fft2/ifft2 pair; the
            # shift is a build-time cost, not a per-call one.
            state["field"] = np.fft.ifftshift(field)
            return state

        if case.algorithm_class == "matrix_dft":
            x = pupil_coords(case)
            u = focus_coords(case)
            # (N_f, N_p) kernel, precomputed. Whether a library hoists this is
            # a real design difference the ledger is meant to expose.
            state["kx"] = np.exp(-2j * np.pi * np.outer(u, x)).astype(case.dtype)
            state["scale"] = np.asarray(case.dx_pupil ** 2, dtype=case.real_dtype)
        else:
            n = max(case.n_fft, case.n_pupil)
            state["n"] = n
            state["offset"] = n // 2 - case.n_pupil // 2
            state["crop"] = n // 2 - case.n_focus // 2
            state["scale"] = np.asarray(case.dx_pupil ** 2, dtype=case.real_dtype)
        return state

    def propagate(self, state):
        case = state["case"]
        if state["cls"] == "free_space":
            out = np.fft.ifft2(np.fft.fft2(state["field"]) * state["H"])
            return np.fft.fftshift(out)

        if state["cls"] == "matrix_dft":
            kx = state["kx"]
            return (kx @ state["field"]) @ kx.T * state["scale"]

        n, off, crop = state["n"], state["offset"], state["crop"]
        big = np.zeros((n, n), dtype=case.dtype)
        big[off:off + case.n_pupil, off:off + case.n_pupil] = state["field"]
        out = np.fft.fftshift(self._fft2(np.fft.ifftshift(big))) * state["scale"]
        nf = case.n_focus
        return out[crop:crop + nf, crop:crop + nf]

    # ------------------------------------------------------ gradient board --
    def supports_gradient(self) -> bool | Unsupported:
        return True

    def build_gradient(self, case: Case, config: Config):
        """Hand-written reverse mode, mirroring prysm's manual adjoint style.

        Kept here so the gradient board's finite-difference gate can be
        validated without prysm installed, and so the expected primitive count
        (2 forward GEMMs + 2 adjoint GEMMs) has a known-good reference.
        """
        noll, theta0, basis = gradient_parameters(case)
        x, u = pupil_coords(case), focus_coords(case)
        kx = np.exp(-2j * np.pi * np.outer(u, x)).astype(np.complex128)
        amp = circular_aperture(case)

        # Target: the unaberrated PSF intensity on the same grid.
        w0 = amp.astype(np.complex128)
        e0 = (kx @ w0) @ kx.T * (case.dx_pupil ** 2)
        return {
            "case": case, "basis": basis, "theta": theta0, "kx": kx, "amp": amp,
            "scale": case.dx_pupil ** 2, "target": np.abs(e0) ** 2,
            "static_opd": opd_waves(case),
        }

    def gradient(self, state):
        case, basis, theta = state["case"], state["basis"], state["theta"]
        kx, amp, scale = state["kx"], state["amp"], state["scale"]
        target = state["target"]

        # ---- forward -------------------------------------------------------
        phs = np.tensordot(theta, basis, axes=(0, 0)) + state["static_opd"]
        w = amp * np.exp(2j * np.pi * phs)
        e = (kx @ w) @ kx.T * scale
        inten = np.abs(e) ** 2
        resid = inten - target
        loss = float(np.mean(resid ** 2))

        # ---- reverse -------------------------------------------------------
        # dL/dI
        ibar = 2.0 * resid / resid.size
        # I = |E|^2  ->  dL/dtheta = 2 Re( sum Ebar * dE/dtheta ), Ebar = Ibar*conj(E)
        ebar = ibar * np.conj(e)
        # E = scale * Kx W Kx^T is linear, so the adjoint is a plain transpose
        # (NOT conjugate transpose -- the conjugation is already carried in Ebar).
        wbar = (kx.T @ ebar) @ kx * scale
        # W = amp*exp(2i pi phs)  ->  dW = W * 2i pi dphs
        phsbar = -4.0 * np.pi * np.imag(wbar * w)
        grad = np.tensordot(basis, phsbar, axes=([1, 2], [0, 1]))
        return loss, grad

    def gradient_theta(self, state) -> np.ndarray:
        return state["theta"]
