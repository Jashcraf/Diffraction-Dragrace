"""Segmented-aperture geometry, and an independent rasterisation of it.

The aperture board asks a different question from every other board in this
suite. Elsewhere the case pins a pupil and the adapters propagate it; here the
*drawing of the pupil is the thing being timed*, so the case cannot hand the
adapters a mask -- it has to hand them a specification and let each one rasterise
it through whatever its own documentation teaches.

That makes "did they draw the same telescope" a real question rather than a
given, and it is what this module exists to answer. `elt_segment_centres`
produces the canonical set of segment centres from the published geometry, and
every adapter selects its own segments by matching against that set rather than
by trusting that two libraries number their segments the same way -- they do not.

GEOMETRY, and one trap in it. The numbers below are the E-ELT Construction
Proposal values (Fig. 3.66):

    outer diameter        39.14634 m
    segment size           1.45 m     VERTEX-TO-VERTEX, not flat-to-flat
    segment gap            0.004 m
    central obscuration    9.4136 m   flat-to-flat of a hexagon, not a circle
    spiders                6 x 0.4 m, at 60i + 30 degrees

The capitalised word is the trap. "1.45 m segments" is usually quoted as a
width, and reading it as flat-to-flat puts the centres 1.454 m apart instead of
1.2597 m, which still draws a plausible-looking segmented pupil -- of the wrong
telescope, roughly 15% too large, and with a segment count that no longer comes
to 798. The centre spacing is flat-to-flat plus gap:

    f2f     = 1.45 * sqrt(3)/2 = 1.2557 m
    spacing = f2f + gap        = 1.2597 m

`_verify_against_published_count` checks the 798 at import of the case rather
than leaving it to be noticed in a figure.
"""
from __future__ import annotations

import numpy as np

#: Peak working set an aperture adapter may reach before it declines the point.
#:
#: Two of these codes build the whole pupil as a stack of per-segment arrays and
#: reduce afterwards, so their cost is n_segments x N^2 rather than N^2 and they
#: fall off a cliff partway up the scan. An adapter that would exceed this
#: declines with a computed figure instead of being killed -- and an OOM kill is
#: not a survivable failure here: the worker dies before writing result.json, so
#: every size that measured cleanly is lost with it.
#:
#: Fixed rather than read from the host, so a scan terminates at the same size on
#: every machine. A curve whose last point depends on how much RAM the runner
#: happened to have is not comparable across hosts, which is the same reason
#: `dragrace report` refuses to merge machine fingerprints. 24 GiB is chosen to
#: fit a 32 GB host with room for the reference rasterisation alongside.
#:
#: Adapters estimate their OWN peak against this, and those estimates are
#: measured rather than derived -- see the dLux adapter, where the naive
#: n_segments x N^2 x 8 figure understated the real peak by 12-17x.
APERTURE_MEMORY_BUDGET_BYTES = 24 * 2**30

#: E-ELT Construction Proposal Fig. 3.66. Metres.
ELT = {
    "outer_diameter_m": 39.14634,
    "segment_vertex_to_vertex_m": 1.45,
    "segment_gap_m": 0.004,
    "central_obscuration_flat_to_flat_m": 9.4136,
    "spider_count": 6,
    "spider_width_m": 0.4,
    "spider_angle_offset_deg": 30.0,
    "rings": 17,
    "n_segments": 798,
}


def _hex_lattice(spacing: float, rings: int) -> np.ndarray:
    """Centres of a hexagonal lattice, nearest-neighbour distance `spacing`.

    Axial coordinates with |q|, |r|, |q+r| <= rings, which is the hexagon-shaped
    patch of the lattice rather than a rhombus -- the rhombus would reach
    2*rings out along one diagonal and pull in segments the trim below is not
    written to remove.
    """
    a1 = np.array([spacing, 0.0])
    a2 = np.array([spacing / 2.0, spacing * np.sqrt(3.0) / 2.0])
    out = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if abs(q + r) > rings:
                continue
            out.append(q * a1 + r * a2)
    return np.array(out)


def _inside_hexagon(x: np.ndarray, y: np.ndarray, flat_to_flat: float,
                    rotation_rad: float = 0.0) -> np.ndarray:
    """Point-in-regular-hexagon, by folding to a single 60-degree wedge.

    Cheaper and more robust than six half-plane tests, and it is the same
    predicate used for the segment shapes and the central obscuration so the two
    cannot disagree about what "a hexagon of this size" means.
    """
    ang = np.arctan2(y, x) - rotation_rad
    rad = np.hypot(x, y)
    # The +pi/6 inside the fold is load-bearing: without it the fold is measured
    # from a vertex rather than from a flat normal, which rotates every hexagon
    # by 30 degrees while keeping its apothem. Individually those hexagons look
    # right and have the right area, so the error survives inspection -- but on
    # a lattice tuned for the unrotated orientation the circumradius (1.1547a)
    # exceeds half the centre spacing, neighbours overlap, and the union comes
    # out ~6% short. That is how it was caught: segment-only fill of 0.665
    # against HCIPy's 0.710.
    folded = np.mod(ang + np.pi / 6.0, np.pi / 3.0) - np.pi / 6.0
    return rad * np.cos(folded) <= flat_to_flat / 2.0


def elt_segment_centres(spec: dict | None = None) -> np.ndarray:
    """The canonical (N, 2) array of ELT segment centres, in metres.

    Reproduces the published layout: a 17-ring hexagonal lattice, with the
    segments over the central obscuration removed and the six pointy corners
    trimmed back to keep the pupil roughly circular.
    """
    s = dict(ELT, **(spec or {}))
    f2f = s["segment_vertex_to_vertex_m"] * np.sqrt(3.0) / 2.0
    spacing = f2f + s["segment_gap_m"]

    c = _hex_lattice(spacing, s["rings"])
    # The lattice above has rows along x; the published pupil is the 30-degree
    # rotation of that. Rotating the centres rather than the lattice keeps the
    # segment shape and the lattice in one consistent frame.
    rot = np.pi / 6.0
    x = c[:, 0] * np.cos(rot) - c[:, 1] * np.sin(rot)
    y = c[:, 0] * np.sin(rot) + c[:, 1] * np.cos(rot)

    # Drop the segments sitting over the central obscuration. Its flats are
    # aligned with the axes (rotation 0) while the segment lattice is rotated
    # 30 degrees; at the other orientation it removes 55 segments instead of 61
    # and the layout comes to 804 rather than the published 798.
    keep = ~_inside_hexagon(x, y, s["central_obscuration_flat_to_flat_m"],
                            rotation_rad=0.0)

    # Trim the six corners. 0.99 of the outer radius is the published figure's
    # cut; at exactly 1.0 the corner segments are tangent and float-point
    # noise decides whether each one survives, which is how a layout ends up
    # differing by a handful of segments between machines.
    r = s["outer_diameter_m"] / 2.0 * 0.99
    c30, s30 = np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)
    keep &= np.abs(y) < r
    keep &= np.abs(c30 * x + s30 * y) < r
    keep &= np.abs(c30 * x - s30 * y) < r

    return np.column_stack([x[keep], y[keep]])


def verify_segment_count(centres: np.ndarray, expected: int = ELT["n_segments"]) -> None:
    if len(centres) != expected:
        raise ValueError(
            f"ELT layout produced {len(centres)} segments, expected {expected}. "
            f"The usual cause is reading segment_vertex_to_vertex_m as a "
            f"flat-to-flat width -- see the module docstring."
        )


def select_for_centres(candidate_centres: np.ndarray, canonical: np.ndarray,
                       tol: float) -> np.ndarray:
    """Indices of `candidate_centres` that coincide with a canonical centre.

    This is how each adapter is held to the same 798 segments. prysm, POPPY and
    lentil all expose a segment index and all three enumerate rings in their own
    order, so an `exclude`/`segmentlist`/`drop` list written for one is wrong for
    the others -- silently, because every one of them still draws a handsome
    segmented pupil. Matching on position instead of on index makes the
    selection a statement about geometry, which is the thing the case pins.
    """
    if len(candidate_centres) == 0:
        return np.zeros(0, dtype=int)
    d = np.linalg.norm(
        np.asarray(candidate_centres)[:, None, :] - canonical[None, :, :], axis=-1)
    return np.flatnonzero(d.min(axis=1) <= tol)


def reference_mask(case, spec: dict | None = None) -> np.ndarray:
    """Independent rasterisation of the case's segmented pupil, for the gate.

    Deliberately naive and deliberately not any adapter's algorithm: every
    segment is tested over the whole grid with the same point-in-hexagon
    predicate used to lay out the centres. Slow, and it does not matter -- this
    is never timed.

    Hard-edged, because the codes under test disagree about antialiasing (HCIPy
    draws two-valued masks, prysm and lentil antialias by default) and there is
    no neutral choice. The gate therefore compares geometry, not edge treatment;
    validate.compare_aperture reports the edge disagreement separately rather
    than folding it into a pass/fail.
    """
    s = dict(ELT, **(spec or {}))
    n = case.n_pupil
    extent = case.pupil.diameter_m
    ax = (np.arange(n) - n / 2.0 + 0.5) * (extent / n)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")

    f2f = s["segment_vertex_to_vertex_m"] * np.sqrt(3.0) / 2.0
    centres = elt_segment_centres(s)

    mask = np.zeros((n, n), dtype=np.float64)
    # Each segment is a hexagon of flat-to-flat f2f, rotated 30 degrees from the
    # lattice rows -- the same rotation applied to the centres above.
    for cx, cy in centres:
        # Restrict to the segment's bounding box; the predicate is exact either
        # way, but testing 798 segments over a 2048^2 grid is otherwise minutes.
        lo_x = np.searchsorted(ax, cx - f2f)
        hi_x = np.searchsorted(ax, cx + f2f) + 1
        lo_y = np.searchsorted(ax, cy - f2f)
        hi_y = np.searchsorted(ax, cy + f2f) + 1
        sub_x = xx[lo_y:hi_y, lo_x:hi_x] - cx
        sub_y = yy[lo_y:hi_y, lo_x:hi_x] - cy
        if sub_x.size == 0:
            continue
        inside = _inside_hexagon(sub_x, sub_y, f2f, rotation_rad=np.pi / 6.0)
        mask[lo_y:hi_y, lo_x:hi_x] = np.maximum(mask[lo_y:hi_y, lo_x:hi_x], inside)

    # Spiders: infinite bars through the centre, subtracted last.
    for i in range(s["spider_count"]):
        ang = np.deg2rad(s["spider_angle_offset_deg"] + 60.0 * i)
        # Distance from the line through the origin at this angle.
        perp = np.abs(-np.sin(ang) * xx + np.cos(ang) * yy)
        along = np.cos(ang) * xx + np.sin(ang) * yy
        mask[(perp <= s["spider_width_m"] / 2.0) & (along >= 0)] = 0.0

    return mask
