"""Canonical grids, apertures and bases.

Every adapter is handed a pupil array built here, rather than building its own.
This is deliberate: aperture antialiasing and Zernike normalisation differ
between these six codes, and those differences would otherwise show up as
accuracy failures or -- worse -- as timing differences, because an antialiased
aperture costs more to render than a hard mask. Removing them from the
measurement is what makes the propagation itself the thing being compared.

The one thing that is *not* pinned is where the samples sit; see CENTRING below.
Two adapters on different centring conventions get the same aperture rule,
evaluated on grids offset by half a sample -- identical cost, identical physics,
different sample positions.

Conventions (see docs/conventions.md):

  * Lengths in the pupil are in units of the pupil diameter D, so the aperture
    has radius 0.5 and sample spacing dx = 1/N_D.
  * Focal-plane coordinates are in units of lambda*F/D, with spacing 1/q.
  * Grids are centred by the case's *centring convention*, `pixel` by default.

CENTRING. Two conventions are in circulation and both are correct
discretisations of the same continuous problem:

  pixel        x[i] = (i - N//2) * dx        the fftshift convention: the origin
                                             lands ON a sample, so an on-axis
                                             PSF peaks in a single pixel.
  interpixel   x[i] = (i - N/2 + 0.5) * dx   the origin lands BETWEEN the middle
                                             samples, so an on-axis PSF is
                                             centred on the four-pixel cross.

Which one a code produces is not a free choice for the benchmark to make.
POPPY's OpticalSystem hard-codes interpixel centring inside `_propagate_mft`,
and no documented knob reaches it; measuring POPPY through the API its own
documentation teaches therefore means measuring it on an interpixel grid.

So the *adapter* declares which convention its output obeys, and the reference,
the injected pupil and the coordinates are all built to match. Every adapter
still receives the identical aperture rule, identical physics and an identically
priced rasterisation -- what differs is only where the samples sit, which is the
one thing a library is entitled to decide. Comparing a code against a reference
on a foreign grid measures the convention mismatch and nothing else: POPPY scores
rel_l2 = 0.28 against a pixel-centred reference and 1.5e-15 against its own.
"""
from __future__ import annotations

import math

import numpy as np

from .case import Case

CENTERINGS = {"pixel", "interpixel"}


def centre_offset(centering: str) -> float:
    """Sample offset applied to every grid built under this convention."""
    if centering not in CENTERINGS:
        raise ValueError(f"unknown centering {centering!r}; expected one of {sorted(CENTERINGS)}")
    return 0.5 if centering == "interpixel" else 0.0


def centering_pair(spec) -> tuple[str, str]:
    """(pupil, focus) from an adapter's declaration.

    A plain string means both planes share a convention, which is the usual
    case. A mapping declares them separately, which is not a hypothetical:
    HCIPy's `make_pupil_grid` is interpixel while its `make_focal_grid` puts a
    sample on the axis, so the two planes genuinely disagree inside one library.
    Forcing a single answer there costs 5.9e-3 of accuracy -- small enough to
    look like a tolerance problem and be "fixed" by loosening the gate, which is
    exactly the mistake this pair exists to prevent.
    """
    if isinstance(spec, str):
        pupil = focus = spec
    else:
        try:
            pupil, focus = spec["pupil"], spec["focus"]
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"grid_centering must be a string or a mapping with 'pupil' and "
                f"'focus' keys, got {spec!r}"
            ) from exc
    centre_offset(pupil)
    centre_offset(focus)
    return pupil, focus


def _axis(n: int, spacing: float, centering: str) -> np.ndarray:
    # n//2 and n/2 differ for odd n; the pixel branch keeps the integer form so
    # that odd-N grids stay symmetric about their centre sample.
    if centering == "interpixel":
        return (np.arange(n, dtype=np.float64) - n / 2.0 + 0.5) * spacing
    centre_offset(centering)                       # validate
    return (np.arange(n, dtype=np.float64) - n // 2) * spacing


# ------------------------------------------------------------- coordinates --
def pupil_coords(case: Case, centering="pixel") -> np.ndarray:
    """1-D pupil coordinate in units of D. Uses the pupil half of `centering`."""
    return _axis(case.n_pupil, case.dx_pupil, centering_pair(centering)[0])


def focus_coords(case: Case, centering="pixel") -> np.ndarray:
    """1-D focal coordinate in lambda*F/D. Uses the focal half of `centering`."""
    return _axis(case.n_focus, case.du_focus, centering_pair(centering)[1])


def pupil_xy(case: Case, centering="pixel") -> tuple[np.ndarray, np.ndarray]:
    """2-D pupil coordinates in units of D, on the case's grid."""
    x = pupil_coords(case, centering)
    return np.meshgrid(x, x, indexing="xy")


def pupil_rho_theta(case: Case, centering="pixel") -> tuple[np.ndarray, np.ndarray]:
    xx, yy = pupil_xy(case, centering)
    return np.hypot(xx, yy), np.arctan2(yy, xx)


# ---------------------------------------------------------------- aperture --
def circular_aperture(case: Case, centering="pixel") -> np.ndarray:
    """Hard-edged unit-transmission circular aperture, radius 0.5 (= D/2).

    Deliberately not antialiased. An antialiased edge is a better optical model
    but it is also a *choice*, and each code makes a different one; pinning the
    hard mask keeps the comparison about propagation rather than about
    rasterisation. Cases needing a grey edge should add an explicit
    `aperture: circular_antialiased` rather than letting adapters differ.
    """
    rho, _ = pupil_rho_theta(case, centering)
    return (rho <= 0.5).astype(np.float64)


def obstructed_aperture(case: Case, centering="pixel") -> np.ndarray:
    """Circular aperture, minus a secondary obscuration, minus spider vanes.

    The phase-retrieval pupil. Hard-edged like every other mask the harness
    injects, and for the same reason: each of these codes antialiases
    differently, and here the difference would land inside a *loss function*
    rather than merely inside an accuracy number -- two codes whose vane edges
    disagree are minimising different functions and their iteration counts stop
    being comparable.

    Why the obstruction and the vanes are physics rather than decoration: an
    unobstructed circular pupil is invariant under rotation, so its PSF cannot
    distinguish the two members of a rotated pair of Zernike modes, and the
    retrieval is degenerate in the astigmatism, coma and trefoil planes. The
    vanes break that. What they do NOT break is the centro-symmetry that every
    element here shares, so the twin ambiguity survives by construction -- OPD
    phi(x) and -phi(-x) give the identical PSF. That is a property of
    single-image phase retrieval, not of this implementation, and it is why the
    gate below is data consistency rather than coefficient recovery.
    """
    ob = case.pupil.obstruction
    if ob is None:
        raise ValueError(
            f"case {case.id!r} asks for an obstructed aperture but carries no "
            f"`obstruction:` block")
    xx, yy = pupil_xy(case, centering)
    rho = np.hypot(xx, yy)

    mask = (rho <= 0.5) & (rho >= 0.5 * ob.secondary_ratio)
    half_width = 0.5 * ob.spider_width_ratio
    for angle in ob.spider_angles_deg:
        a = math.radians(angle)
        # Perpendicular distance from the line through the origin at this angle,
        # and the signed distance along it.
        perp = np.abs(-math.sin(a) * xx + math.cos(a) * yy)
        vane = perp <= half_width
        if ob.spider_span == "radius":
            vane &= (math.cos(a) * xx + math.sin(a) * yy) >= 0.0
        mask &= ~vane
    return mask.astype(np.float64)


def aperture_mask(case: Case, centering="pixel") -> np.ndarray:
    """The case's amplitude mask, whichever kind of pupil it declares.

    A segmented pupil has no entry here on purpose: on that board the drawing is
    the measurement, so the case hands the adapters a specification and each one
    rasterises it itself (dragrace.apertures).
    """
    kind = case.pupil.aperture
    if kind == "circular":
        return circular_aperture(case, centering)
    if kind == "obstructed":
        return obstructed_aperture(case, centering)
    raise ValueError(
        f"no harness rasterisation for aperture {kind!r}; a segmented pupil is "
        f"drawn by the adapter under test, not injected")


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


def zernike_basis(case: Case, noll_indices: list[int], centering="pixel") -> np.ndarray:
    """(P, N_p, N_p) Zernike basis, numerically normalised to unit RMS over the
    aperture and zero outside it.

    The analytic Noll normalisation gives unit RMS over the *continuous* unit
    disk; on a discrete grid it is only approximately unit RMS. Renormalising
    numerically makes "waves RMS" mean exactly that, so a coefficient is
    comparable across grid sizes -- which matters because N is a swept axis.

    Normalised over the case's OWN mask, which for an obstructed pupil is the
    annulus minus the vanes rather than the full disk. The modes are then unit
    RMS over the area that actually transmits light, so "0.05 waves RMS" means
    the same physical wavefront error whether or not the case carries a
    secondary -- and it keeps the retrieval's parameter scaling independent of
    the obscuration ratio.
    """
    rho, theta = pupil_rho_theta(case, centering)
    mask = aperture_mask(case, centering) > 0
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


def opd_waves(case: Case, centering="pixel") -> np.ndarray:
    """Static aberration OPD in waves, from the case's Zernike coefficients."""
    ab = case.pupil.aberration
    if not ab.coefficients:
        return np.zeros((case.n_pupil, case.n_pupil), dtype=np.float64)
    if ab.convention != "noll_rms_waves":
        raise ValueError(f"unsupported aberration convention {ab.convention!r}")
    basis = zernike_basis(case, ab.noll_indices, centering)
    return np.tensordot(np.asarray(ab.values), basis, axes=(0, 0))


# ------------------------------------------------------------- pupil field --
def pupil_field(case: Case, opd: np.ndarray | None = None, centering="pixel") -> np.ndarray:
    """Complex pupil field: amplitude mask times phasor, in the case dtype.

    OPD is carried in waves so the phasor is exp(2j*pi*opd) with no wavelength
    factor -- this keeps the harness free of the metres-vs-microns unit slips
    that are otherwise a recurring source of cross-code disagreement.
    """
    amp = aperture_mask(case, centering)
    if opd is None:
        opd = opd_waves(case, centering)
    field = amp * np.exp(2j * np.pi * opd)
    return field.astype(case.dtype)


def gradient_parameters(case: Case, centering="pixel") -> tuple[list[int], np.ndarray, np.ndarray]:
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
    return noll, theta0, zernike_basis(case, noll, centering)
