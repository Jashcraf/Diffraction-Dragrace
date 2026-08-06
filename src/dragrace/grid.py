"""Canonical grids, apertures and bases.

Every adapter is handed the *identical* pupil array built here, rather than
building its own. This is deliberate: aperture antialiasing, grid centring and
Zernike normalisation all differ between these six codes, and those differences
would otherwise show up as accuracy failures or -- worse -- as timing
differences, because an antialiased aperture costs more to render than a hard
mask. Removing them from the measurement is what makes the propagation itself
the thing being compared.

Conventions (see docs/conventions.md):

  * Lengths in the pupil are in units of the pupil diameter D, so the aperture
    has radius 0.5 and sample spacing dx = 1/N_D.
  * Focal-plane coordinates are in units of lambda*F/D, with spacing 1/q.
  * Both grids are centred at index N//2 -- the fftshift convention. For even
    N this leaves the grid asymmetric by one sample, which is standard, and is
    consistent between the MFT and FFT paths so the two agree to roundoff.
"""
from __future__ import annotations

import math

import numpy as np

from .case import Case


# ------------------------------------------------------------- coordinates --
def pupil_coords(case: Case) -> np.ndarray:
    """1-D pupil coordinate in units of D."""
    n = case.n_pupil
    return (np.arange(n, dtype=np.float64) - n // 2) * case.dx_pupil


def focus_coords(case: Case) -> np.ndarray:
    """1-D focal coordinate in units of lambda*F/D."""
    n = case.n_focus
    return (np.arange(n, dtype=np.float64) - n // 2) * case.du_focus


def pupil_rho_theta(case: Case) -> tuple[np.ndarray, np.ndarray]:
    x = pupil_coords(case)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return np.hypot(xx, yy), np.arctan2(yy, xx)


# ---------------------------------------------------------------- aperture --
def circular_aperture(case: Case) -> np.ndarray:
    """Hard-edged unit-transmission circular aperture, radius 0.5 (= D/2).

    Deliberately not antialiased. An antialiased edge is a better optical model
    but it is also a *choice*, and each code makes a different one; pinning the
    hard mask keeps the comparison about propagation rather than about
    rasterisation. Cases needing a grey edge should add an explicit
    `aperture: circular_antialiased` rather than letting adapters differ.
    """
    rho, _ = pupil_rho_theta(case)
    return (rho <= 0.5).astype(np.float64)


# ----------------------------------------------------------------- Zernike --
def noll_to_nm(j: int) -> tuple[int, int]:
    """Noll index -> (n, m). j=1 is piston."""
    if j < 1:
        raise ValueError("Noll indices start at 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2))
    return n, m


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        out += c * rho ** (n - 2 * k)
    return out


def zernike_basis(case: Case, noll_indices: list[int]) -> np.ndarray:
    """(P, N_p, N_p) Zernike basis, numerically normalised to unit RMS over the
    aperture and zero outside it.

    The analytic Noll normalisation gives unit RMS over the *continuous* unit
    disk; on a discrete grid it is only approximately unit RMS. Renormalising
    numerically makes "waves RMS" mean exactly that, so a coefficient is
    comparable across grid sizes -- which matters because N is a swept axis.
    """
    rho, theta = pupil_rho_theta(case)
    mask = circular_aperture(case) > 0
    rho_unit = rho / 0.5                       # radius 0.5 -> unit disk

    modes = np.zeros((len(noll_indices), case.n_pupil, case.n_pupil), dtype=np.float64)
    for i, j in enumerate(noll_indices):
        n, m = noll_to_nm(j)
        R = _zernike_radial(n, abs(m), rho_unit)
        if m == 0:
            z = R
        elif m > 0:
            z = R * np.cos(m * theta)
        else:
            z = R * np.sin(abs(m) * theta)
        z = np.where(mask, z, 0.0)
        if j != 1:                             # piston has no meaningful RMS scaling
            z -= z[mask].mean()
        rms = float(np.sqrt(np.mean(z[mask] ** 2)))
        if rms > 0:
            z /= rms
        modes[i] = np.where(mask, z, 0.0)
    return modes


def opd_waves(case: Case) -> np.ndarray:
    """Static aberration OPD in waves, from the case's Zernike coefficients."""
    ab = case.pupil.aberration
    if not ab.coefficients:
        return np.zeros((case.n_pupil, case.n_pupil), dtype=np.float64)
    if ab.convention != "noll_rms_waves":
        raise ValueError(f"unsupported aberration convention {ab.convention!r}")
    basis = zernike_basis(case, ab.noll_indices)
    return np.tensordot(np.asarray(ab.values), basis, axes=(0, 0))


# ------------------------------------------------------------- pupil field --
def pupil_field(case: Case, opd: np.ndarray | None = None) -> np.ndarray:
    """Complex pupil field: amplitude mask times phasor, in the case dtype.

    OPD is carried in waves so the phasor is exp(2j*pi*opd) with no wavelength
    factor -- this keeps the harness free of the metres-vs-microns unit slips
    that are otherwise a recurring source of cross-code disagreement.
    """
    amp = circular_aperture(case)
    if opd is None:
        opd = opd_waves(case)
    field = amp * np.exp(2j * np.pi * opd)
    return field.astype(case.dtype)


def gradient_parameters(case: Case) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(noll_indices, theta0, basis) for a gradient case.

    theta0 is drawn from a seeded RNG so every adapter differentiates at the
    same point -- gradients are point-valued and comparing them at different
    theta would be meaningless.
    """
    p = case.parameters
    if p is None:
        raise ValueError(f"case {case.id!r} has no parameters block")
    if p.basis != "zernike_noll":
        raise ValueError(f"unsupported gradient basis {p.basis!r}")
    noll = list(range(p.first_noll, p.first_noll + p.count))
    rng = np.random.default_rng(p.seed)
    theta0 = rng.normal(0.0, p.amplitude_waves_rms, size=p.count)
    return noll, theta0, zernike_basis(case, noll)
