"""Shared, package-agnostic definition of the phase-retrieval benchmark problem.

Everything in here is pure ``numpy`` so that it can be imported unchanged from
inside every propagator's conda environment (poppy, hcipy, prysm, dLux).  The
point is that *all* packages solve the identical problem:

    * the same Subaru-like asymmetric pupil amplitude,
    * the same Zernike (Noll) modal basis for the unknown phase,
    * the same ground-truth aberration coefficients,
    * the same focal-plane sampling.

A single, in-focus point-spread function (PSF) is enough to recover the phase
here because the aperture is *not* point-symmetric.  For a centro-symmetric
pupil, |FT|^2 cannot distinguish phi(x, y) from phi(-x, -y) (the classic
"twin-image" ambiguity), which is why phase retrieval usually needs focus
diversity.  Breaking the pupil symmetry -- as the offset Subaru spider does --
lifts that degeneracy, so no defocused frames are required.

The physical numbers are loosely those of the Subaru Telescope (D = 8.2 m,
observing in H band), but the benchmark only cares that every package sees the
same arrays and the same focal sampling.
"""

import math

import numpy as np

# --------------------------------------------------------------------------
# Physical / sampling parameters (SI units, metres and radians throughout)
# --------------------------------------------------------------------------
WAVELENGTH = 1.65e-6        # H band, metres
DIAMETER = 8.2              # Subaru primary diameter, metres
EFL = 1.0                   # effective focal length, metres (arbitrary scale)

# Focal-plane sampling, shared by every package.
FOCAL_Q = 3                 # samples per (lambda / D) resolution element
FOCAL_PIXELS = 64           # focal-plane grid is FOCAL_PIXELS x FOCAL_PIXELS

# Zernike modal basis for the unknown phase.  We skip piston/tip/tilt (Noll
# 1-3): piston is unobservable and tip/tilt only translate the PSF, so they are
# not interesting aberrations to retrieve.
NOLL_START = 4              # first Noll index (defocus)
N_MODES = 11               # Noll 4..14 inclusive


def focal_pixel_scale_radians():
    """Angular size of one focal-plane pixel, in radians."""
    return WAVELENGTH / (FOCAL_Q * DIAMETER)


def truth_coeffs(seed=1234):
    """Ground-truth Zernike coefficients (radians of phase at ``WAVELENGTH``).

    Kept deliberately small (RMS well below 1 rad, peak-to-valley below pi) so
    that there is no 2*pi phase wrapping and a single PSF has an unambiguous
    global minimum -- the retrieval should drive the cost essentially to zero
    for every package.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1.0, 1.0, size=N_MODES)
    # Taper the higher-order modes and scale to a modest total aberration.
    taper = 1.0 / np.arange(1, N_MODES + 1)
    c = c * taper
    c *= 0.35 / np.sqrt(np.sum(c ** 2))   # ~0.35 rad RMS total
    return c.astype(np.float64)


# --------------------------------------------------------------------------
# Subaru-like asymmetric pupil
# --------------------------------------------------------------------------
def subaru_pupil(n, secondary_ratio=0.30, spider_width_frac=0.012,
                 spider_offset_frac=0.10):
    """Return an ``(n, n)`` float64 amplitude mask for a Subaru-like pupil.

    The mask is a filled circle with a central obscuration and four spider
    vanes.  Crucially the vanes are *offset* from the centre (all shifted in the
    same rotational sense), so the amplitude is not invariant under
    (x, y) -> (-x, -y).  That asymmetry is what makes single-image phase
    retrieval well posed.

    Parameters
    ----------
    n : int
        Number of pixels across the pupil.
    secondary_ratio : float
        Central obscuration diameter as a fraction of the primary.
    spider_width_frac : float
        Spider vane half-width as a fraction of the primary radius.
    spider_offset_frac : float
        Perpendicular offset of each vane from the centre, as a fraction of the
        primary radius.  This is the term that breaks point symmetry.
    """
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)

    amp = np.ones((n, n), dtype=np.float64)
    amp[r > 1.0] = 0.0                     # primary aperture edge
    amp[r < secondary_ratio] = 0.0         # central obscuration

    w = spider_width_frac
    d = spider_offset_frac
    # Four vanes forming a cross whose intersection sits at (d, d) instead of the
    # pupil centre -- exactly the off-centre spider attachment of the real Subaru
    # pupil.  A cross centred off-origin maps under (x, y) -> (-x, -y) onto a
    # cross centred at (-d, -d), so it is not present in the original mask: the
    # amplitude is genuinely non-centro-symmetric, which is what makes a single
    # in-focus PSF sufficient for phase retrieval.
    ox, oy = d, d
    spider = (
        ((np.abs(Y - oy) < w) & (X > ox)) |   # +x vane from off-centre hub
        ((np.abs(Y - oy) < w) & (X < ox)) |   # -x vane
        ((np.abs(X - ox) < w) & (Y > oy)) |   # +y vane
        ((np.abs(X - ox) < w) & (Y < oy))     # -y vane
    )
    amp[spider] = 0.0
    return amp


# --------------------------------------------------------------------------
# Zernike (Noll) modal basis
# --------------------------------------------------------------------------
def _noll_to_nm(j):
    """Convert a 1-based Noll index to (n, m) radial/azimuthal orders."""
    if j < 1:
        raise ValueError("Noll index must be >= 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** (j % 2) * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _zernike_radial(n, m, rho):
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * math.factorial(n - k)
        den = (math.factorial(k)
               * math.factorial((n + m) // 2 - k)
               * math.factorial((n - m) // 2 - k))
        R += (num / den) * rho ** (n - 2 * k)
    return R


def _zernike(j, rho, theta):
    """Noll-normalised Zernike polynomial number ``j`` on the unit disk."""
    n, m = _noll_to_nm(j)
    norm = np.sqrt(n + 1)
    if m == 0:
        return norm * _zernike_radial(n, 0, rho)
    elif m > 0:
        return norm * np.sqrt(2) * _zernike_radial(n, m, rho) * np.cos(m * theta)
    else:
        return norm * np.sqrt(2) * _zernike_radial(n, -m, rho) * np.sin(-m * theta)


def zernike_basis(n, start=NOLL_START, nmodes=N_MODES):
    """Return an ``(nmodes, n, n)`` float64 array of Zernike modes.

    The modes are evaluated on the unit disk inscribed in the ``n x n`` grid and
    are zero outside it.  The phase used everywhere in the benchmark is
    ``phase = sum_k coeffs[k] * basis[k]`` (radians).
    """
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    inside = rho <= 1.0
    modes = np.zeros((nmodes, n, n), dtype=np.float64)
    for k in range(nmodes):
        z = _zernike(start + k, rho, theta)
        z[~inside] = 0.0
        modes[k] = z
    return modes


def phase_from_coeffs(coeffs, basis):
    """Build the pupil phase (radians) from modal coefficients."""
    return np.tensordot(np.asarray(coeffs), basis, axes=(0, 0))


# --------------------------------------------------------------------------
# Convenience bundle
# --------------------------------------------------------------------------
def make_problem(n):
    """Return a dict with every array/scalar a package needs for the benchmark."""
    amp = subaru_pupil(n)
    basis = zernike_basis(n)
    coeffs = truth_coeffs()
    return {
        "n": n,
        "amp": amp,
        "basis": basis,
        "truth_coeffs": coeffs,
        "truth_phase": phase_from_coeffs(coeffs, basis),
        "wavelength": WAVELENGTH,
        "diameter": DIAMETER,
        "efl": EFL,
        "pupil_dx": DIAMETER / n,
        "focal_pixels": FOCAL_PIXELS,
        "focal_q": FOCAL_Q,
        "focal_pixel_scale_rad": focal_pixel_scale_radians(),
        "n_modes": N_MODES,
        "noll_start": NOLL_START,
    }


if __name__ == "__main__":
    # Quick visual / sanity check.
    import matplotlib.pyplot as plt
    p = make_problem(256)
    print("pupil fill fraction:", p["amp"].mean())
    print("truth coeffs (rad):", np.round(p["truth_coeffs"], 4))
    print("truth phase RMS (rad):",
          np.sqrt(np.mean(p["truth_phase"][p["amp"] > 0] ** 2)))
    # Confirm the aperture is genuinely asymmetric.
    asym = np.abs(p["amp"] - p["amp"][::-1, ::-1]).sum() / p["amp"].sum()
    print("point-asymmetry fraction:", asym)
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(p["amp"], cmap="gray"); ax[0].set_title("Subaru-like pupil")
    ax[1].imshow(p["truth_phase"], cmap="RdBu"); ax[1].set_title("truth phase")
    ax[2].imshow(p["basis"][0], cmap="RdBu"); ax[2].set_title("Noll %d" % NOLL_START)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig("results/problem_setup.png", dpi=110)
    print("saved results/problem_setup.png")
