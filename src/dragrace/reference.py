"""Independent reference implementations.

Nothing here is an adapter and nothing here is timed. These exist purely to
answer "did the adapter compute the right thing", and they are written
independently of the numpy_baseline adapter on purpose -- a reference that
shares code with the thing it validates is not a reference.

Two levels:

  analytic_airy   closed form for an unaberrated circular pupil. The strongest
                  check available, because it depends on no discretisation.
  internal_mft    a direct, deliberately naive double-precision matrix DFT for
                  aberrated cases where no closed form exists.
"""
from __future__ import annotations

import numpy as np

from dataclasses import replace

from .case import Case
from .grid import circular_aperture, focus_coords, opd_waves, pupil_coords


def airy_field(case: Case, centering="pixel") -> np.ndarray:
    """Analytic focal field of an unaberrated circular pupil, on the case grid.

    For a disk of diameter D=1 with unit transmission, the Fourier transform is

        E(r) = (pi/4) * 2 * J1(pi r) / (pi r),      r in lambda*F/D

    which is exactly what a correctly normalised propagator returns when the
    pupil field is the unit-amplitude mask and the transform carries dx^2.
    """
    from scipy.special import j1                      # scipy is a dev/test dep only

    u = focus_coords(case, centering)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r = np.hypot(uu, vv)
    x = np.pi * r
    with np.errstate(invalid="ignore", divide="ignore"):
        core = np.where(x == 0, 1.0, 2.0 * j1(x) / np.where(x == 0, 1.0, x))
    return ((np.pi / 4.0) * core).astype(np.complex128)


def reference_mft(case: Case, centering="pixel") -> np.ndarray:
    """Direct matrix DFT in float64, independent of the baseline adapter.

    Written for transparency rather than speed -- it is the definition of the
    transform this benchmark means, spelled out.
    """
    x = pupil_coords(case, centering)
    u = focus_coords(case, centering)
    field = (circular_aperture(case, centering)
             * np.exp(2j * np.pi * opd_waves(case, centering))).astype(np.complex128)

    # E_f[v,u] = sum_{y,x} E_p[y,x] exp(-2i pi (x u + y v)) dx dy
    kx = np.exp(-2j * np.pi * np.outer(u, x))         # (N_f, N_p)
    tmp = kx @ field                                   # (N_f, N_p)
    out = tmp @ kx.T                                   # (N_f, N_f)
    return out * (case.dx_pupil ** 2)


def free_space_transfer_function(case: Case, kind: str) -> np.ndarray:
    """The two free-space kernels this suite distinguishes, side by side.

        angular_spectrum   H = exp(i k z sqrt(1 - (lam f)^2))     exact
        fresnel_tf         H = exp(i k z) exp(-i pi lam z f^2)    paraxial

    The second is the binomial expansion of the first truncated after the
    quadratic term. They are not interchangeable: on the shipped 2-inch-at-1-m
    geometry they disagree by 4.07e-6, which is four orders above the gate. A
    case declares which one it means through its algorithm_class, so a code
    implementing the paraxial kernel is measured against the paraxial reference
    and passes on its own terms rather than being marked wrong for a choice its
    documentation is explicit about.

    Evanescent components are zeroed in the exact form rather than allowed to
    grow; the paraxial form has no evanescent cutoff to apply, which is one of
    the ways the approximation shows itself.
    """
    lam = case.wavelength_m
    z = case.propagation.distance_m
    f = np.fft.fftfreq(case.n_pupil, d=case.dx_pupil_m)
    fx, fy = np.meshgrid(f, f, indexing="xy")
    f2 = fx ** 2 + fy ** 2

    if kind == "angular_spectrum":
        arg = 1.0 - lam ** 2 * f2
        kz = 2.0 * np.pi / lam * np.sqrt(np.abs(arg))
        return np.where(arg >= 0.0, np.exp(1j * kz * z), 0.0)
    if kind in ("fresnel_tf", "fresnel_ir"):
        return np.exp(2j * np.pi * z / lam) * np.exp(-1j * np.pi * lam * z * f2)
    raise ValueError(f"no free-space kernel for algorithm_class {kind!r}")


def reference_free_space(case: Case, centering="pixel") -> np.ndarray:
    """Propagate the case's aperture by the kernel its algorithm_class names.

    The ifftshift/fftshift pair is not decoration: the harness centres its grids
    at index N//2 while numpy.fft puts the origin at index 0, and getting that
    wrong produces a checkerboard sign pattern rather than an obvious failure.
    """
    field = (circular_aperture(case, centering)
             * np.exp(2j * np.pi * opd_waves(case, centering))).astype(np.complex128)
    h = free_space_transfer_function(case, case.algorithm_class)
    spectrum = np.fft.fft2(np.fft.ifftshift(field))
    return np.fft.fftshift(np.fft.ifft2(spectrum * h))


def reference_angular_spectrum(case: Case, centering="pixel") -> np.ndarray:
    """Exact angular-spectrum reference. Kept as a name of its own because a
    case may ask for it explicitly through accuracy.reference."""
    return reference_free_space(replace(case, algorithm_class="angular_spectrum"), centering)


def reference_fresnel_tf(case: Case, centering="pixel") -> np.ndarray:
    """Paraxial Fresnel transfer-function reference."""
    return reference_free_space(replace(case, algorithm_class="fresnel_tf"), centering)


def reference_field(case: Case, centering="pixel") -> np.ndarray:
    """Whichever reference the case asked for, on the adapter's grid.

    `centering` is the adapter's declared convention, not a free parameter: a
    code is compared against the reference for the grid it actually produces.
    Handing it the other one measures the convention mismatch and nothing else.
    """
    which = case.accuracy.reference
    if which == "analytic_airy":
        return airy_field(case, centering)
    if which == "internal_mft":
        return reference_mft(case, centering)
    if which == "internal_angular_spectrum":
        return reference_angular_spectrum(case, centering)
    if which == "internal_fresnel_tf":
        return reference_fresnel_tf(case, centering)
    raise ValueError(f"unknown accuracy.reference {which!r}")


# --------------------------------------------------------- gradient board ---
def loss_and_reference_gradient(case: Case, theta: np.ndarray,
                                centering="pixel") -> tuple[float, np.ndarray]:
    """Reference loss and its gradient by central differences.

    This is the ground truth the gradient board is gated against. O(2P)
    propagations, so it is a correctness gate run once at small N -- never part
    of a timed measurement.
    """
    from .grid import gradient_parameters

    _, _, basis = gradient_parameters(case, centering)
    target = np.abs(_forward_field(case, np.zeros_like(theta), basis, centering)) ** 2

    def loss(t: np.ndarray) -> float:
        inten = np.abs(_forward_field(case, t, basis, centering)) ** 2
        return float(np.mean((inten - target) ** 2))

    # h is scaled to the parameter amplitude: too small and roundoff dominates,
    # too large and truncation does. 1e-6 waves against ~5e-2 waves of signal
    # sits comfortably between the two for float64.
    h = 1e-6
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        grad[i] = (loss(tp) - loss(tm)) / (2 * h)
    return loss(theta), grad


def _forward_field(case: Case, theta: np.ndarray, basis: np.ndarray,
                   centering="pixel") -> np.ndarray:
    """The gradient board's forward model, in float64, independent of adapters."""
    x = pupil_coords(case, centering)
    u = focus_coords(case, centering)
    opd = np.tensordot(theta, basis, axes=(0, 0))
    field = circular_aperture(case, centering) * np.exp(2j * np.pi * opd)
    kx = np.exp(-2j * np.pi * np.outer(u, x))
    return (kx @ field) @ kx.T * (case.dx_pupil ** 2)
