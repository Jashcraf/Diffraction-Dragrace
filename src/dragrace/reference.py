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

from .case import Case
from .grid import circular_aperture, focus_coords, opd_waves, pupil_coords


def airy_field(case: Case) -> np.ndarray:
    """Analytic focal field of an unaberrated circular pupil, on the case grid.

    For a disk of diameter D=1 with unit transmission, the Fourier transform is

        E(r) = (pi/4) * 2 * J1(pi r) / (pi r),      r in lambda*F/D

    which is exactly what a correctly normalised propagator returns when the
    pupil field is the unit-amplitude mask and the transform carries dx^2.
    """
    from scipy.special import j1                      # scipy is a dev/test dep only

    u = focus_coords(case)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r = np.hypot(uu, vv)
    x = np.pi * r
    with np.errstate(invalid="ignore", divide="ignore"):
        core = np.where(x == 0, 1.0, 2.0 * j1(x) / np.where(x == 0, 1.0, x))
    return ((np.pi / 4.0) * core).astype(np.complex128)


def reference_mft(case: Case) -> np.ndarray:
    """Direct matrix DFT in float64, independent of the baseline adapter.

    Written for transparency rather than speed -- it is the definition of the
    transform this benchmark means, spelled out.
    """
    x = pupil_coords(case)
    u = focus_coords(case)
    field = (circular_aperture(case) * np.exp(2j * np.pi * opd_waves(case))).astype(np.complex128)

    # E_f[v,u] = sum_{y,x} E_p[y,x] exp(-2i pi (x u + y v)) dx dy
    kx = np.exp(-2j * np.pi * np.outer(u, x))         # (N_f, N_p)
    tmp = kx @ field                                   # (N_f, N_p)
    out = tmp @ kx.T                                   # (N_f, N_f)
    return out * (case.dx_pupil ** 2)


def reference_field(case: Case) -> np.ndarray:
    """Whichever reference the case asked for."""
    which = case.accuracy.reference
    if which == "analytic_airy":
        return airy_field(case)
    if which == "internal_mft":
        return reference_mft(case)
    raise ValueError(f"unknown accuracy.reference {which!r}")


# --------------------------------------------------------- gradient board ---
def loss_and_reference_gradient(case: Case, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Reference loss and its gradient by central differences.

    This is the ground truth the gradient board is gated against. O(2P)
    propagations, so it is a correctness gate run once at small N -- never part
    of a timed measurement.
    """
    from .grid import gradient_parameters

    _, _, basis = gradient_parameters(case)
    target = np.abs(_forward_field(case, np.zeros_like(theta), basis)) ** 2

    def loss(t: np.ndarray) -> float:
        inten = np.abs(_forward_field(case, t, basis)) ** 2
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


def _forward_field(case: Case, theta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """The gradient board's forward model, in float64, independent of adapters."""
    x = pupil_coords(case)
    u = focus_coords(case)
    opd = np.tensordot(theta, basis, axes=(0, 0))
    field = circular_aperture(case) * np.exp(2j * np.pi * opd)
    kx = np.exp(-2j * np.pi * np.outer(u, x))
    return (kx @ field) @ kx.T * (case.dx_pupil ** 2)
