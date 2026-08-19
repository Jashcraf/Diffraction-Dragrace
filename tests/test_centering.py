"""Grid centring: the convention an adapter declares, and what it costs to get wrong.

Two of the six codes (POPPY, dLux) centre their PSF between the middle pixels
and offer no documented way out of it. The harness therefore builds the
reference, the injected pupil and the coordinates on the convention the adapter
declares. These tests pin the two properties that makes the scheme trustworthy:
the grids really are half a sample apart, and the mismatch really is catastrophic
-- so a missing or wrong declaration fails loudly rather than shaving accuracy.
"""
import numpy as np
import pytest

from dragrace import validate
from dragrace.adapter import Adapter
from dragrace.case import Case
from dragrace.grid import (CENTERINGS, centre_offset, circular_aperture, focus_coords,
                           pupil_coords)
from dragrace.reference import reference_field

CASE = Case.from_yaml("cases/pupil_to_focus/mft_n1024_q4.yaml")


def _small():
    """A 64-sample version of the reference case, for tests that propagate."""
    d = {
        "id": "small", "kind": "pupil_to_focus", "algorithm_class": "matrix_dft",
        "wavelength_m": CASE.wavelength_m,
        "pupil": {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 64},
        "output": {"focal_length_m": 1.0, "samples_per_lambda_f_d": 4.0,
                   "extent_lambda_f_d": 8.0},
        "accuracy": {"reference": "internal_mft", "max_rel_l2": 1e-10},
    }
    return Case.from_dict(d)


def test_pixel_grid_has_a_sample_on_the_origin():
    x = pupil_coords(CASE, "pixel")
    assert x[CASE.n_pupil // 2] == 0.0
    u = focus_coords(CASE, "pixel")
    assert u[CASE.n_focus // 2] == 0.0


def test_interpixel_grid_straddles_the_origin():
    u = focus_coords(CASE, "interpixel")
    n = CASE.n_focus
    assert u[n // 2 - 1] == pytest.approx(-u[n // 2])
    assert 0.0 not in u


def test_the_two_conventions_differ_by_exactly_half_a_sample():
    a = pupil_coords(CASE, "pixel")
    b = pupil_coords(CASE, "interpixel")
    assert np.allclose(b - a, 0.5 * CASE.dx_pupil)


def test_default_is_pixel():
    assert np.array_equal(pupil_coords(CASE), pupil_coords(CASE, "pixel"))
    assert centre_offset("pixel") == 0.0
    assert centre_offset("interpixel") == 0.5


def test_unknown_centering_is_rejected():
    with pytest.raises(ValueError, match="unknown centering"):
        pupil_coords(CASE, "middle-ish")


@pytest.mark.parametrize("centering", sorted(CENTERINGS))
def test_aperture_is_the_same_hard_edged_rule_either_way(centering):
    """Same rule, different sample positions -- and therefore the same cost.

    If one convention produced a greyscale edge and the other a hard one, the
    scheme would be smuggling a rasterisation difference into a propagation
    comparison, which is exactly what docs/conventions.md forbids.
    """
    amp = circular_aperture(CASE, centering)
    assert set(np.unique(amp)) <= {0.0, 1.0}
    # Area agrees with pi*r^2 to better than the perimeter's worth of samples.
    assert amp.sum() == pytest.approx(np.pi * 0.25 * CASE.n_across**2, rel=1e-3)


def test_matching_convention_agrees_to_roundoff():
    """A propagator on its own grid must land on the reference exactly."""
    case = _small()
    for centering in sorted(CENTERINGS):
        x = pupil_coords(case, centering)
        u = focus_coords(case, centering)
        field = circular_aperture(case, centering).astype(np.complex128)
        k = np.exp(-2j * np.pi * np.outer(u, x))
        out = (k @ field) @ k.T * case.dx_pupil**2
        c = validate.compare(out, reference_field(case, centering), case)
        assert c.gate == "pass" and c.rel_l2 < 1e-13, centering
        assert c.peak_offset_px == (0, 0)


def test_mismatched_convention_fails_loudly():
    """The whole scheme rests on this: an undeclared mismatch cannot pass.

    Measured against POPPY: 0.28 rel_l2 and a one-pixel peak offset. A gate that
    merely degraded would let a wrong declaration through as "slightly less
    accurate", which is how a half-pixel error becomes a published number.
    """
    case = _small()
    x = pupil_coords(case, "interpixel")
    u = focus_coords(case, "interpixel")
    field = circular_aperture(case, "interpixel").astype(np.complex128)
    k = np.exp(-2j * np.pi * np.outer(u, x))
    out = (k @ field) @ k.T * case.dx_pupil**2

    c = validate.compare(out, reference_field(case, "pixel"), case)
    assert c.gate == "fail"
    assert c.rel_l2 > 0.1
    assert c.peak_offset_px != (0, 0)


def test_adapters_declare_a_valid_convention():
    """Including the default: an adapter that says nothing gets pixel.

    Accepts either form -- a bare string for the common case, or a per-plane
    mapping for a library whose two planes disagree.
    """
    from dragrace import adapter as adapter_mod
    from dragrace.grid import centering_pair

    assert Adapter.grid_centering == "pixel"
    adapter_mod.discover("adapters")
    for name in adapter_mod.available():
        ad = adapter_mod.get(name)
        pupil, focus = centering_pair(ad.grid_centering)     # raises if malformed
        assert {pupil, focus} <= CENTERINGS, f"{name}: {ad.grid_centering!r}"


def test_poppy_and_dlux_declare_interpixel():
    """Pinned because it is not a preference -- both libraries fix it internally.

    POPPY hard-codes MatrixFourierTransform(centering='ADJUSTABLE') inside
    poppy_core._propagate_mft. If a future version exposes the choice, this test
    should be revisited deliberately rather than quietly relaxed.
    """
    from dragrace import adapter as adapter_mod

    adapter_mod.discover("adapters")
    for name in ("poppy", "dlux"):
        assert adapter_mod.get(name).grid_centering == "interpixel"


# ------------------------------------------------------- per-plane centring --
def test_centering_pair_accepts_a_string_for_both_planes():
    from dragrace.grid import centering_pair

    assert centering_pair("pixel") == ("pixel", "pixel")
    assert centering_pair("interpixel") == ("interpixel", "interpixel")


def test_centering_pair_accepts_a_mapping():
    from dragrace.grid import centering_pair

    assert centering_pair({"pupil": "interpixel", "focus": "pixel"}) \
        == ("interpixel", "pixel")


@pytest.mark.parametrize("spec", [
    {"pupil": "interpixel"},                       # missing focus
    {"pupil": "interpixel", "focus": "middle"},    # bad value
    ("interpixel", "pixel"),                       # tuple is not a mapping
])
def test_centering_pair_rejects_malformed_specs(spec):
    from dragrace.grid import centering_pair

    with pytest.raises(ValueError):
        centering_pair(spec)


def test_mixed_convention_selects_per_plane():
    """The pupil grid takes the pupil half, the focal grid the focal half."""
    mixed = {"pupil": "interpixel", "focus": "pixel"}
    assert np.array_equal(pupil_coords(CASE, mixed), pupil_coords(CASE, "interpixel"))
    assert np.array_equal(focus_coords(CASE, mixed), focus_coords(CASE, "pixel"))


def test_hcipy_declares_the_mixed_convention():
    """HCIPy's make_pupil_grid is interpixel while make_focal_grid is not.

    Declaring one convention for both planes costs rel_l2 = 5.9e-3 -- small
    enough to be mistaken for a tolerance problem rather than a grid error,
    which is precisely why it is pinned here.
    """
    from dragrace import adapter as adapter_mod

    adapter_mod.discover("adapters")
    assert adapter_mod.get("hcipy").grid_centering == {
        "pupil": "interpixel", "focus": "pixel"}


def test_mixed_mismatch_is_not_silently_small():
    """A half-pixel error in one plane only must still fail the gate.

    It does not look like the 0.28 of a full mismatch -- the peak stays put and
    only a residual phase ramp survives -- so this is the case most likely to be
    waved through as noise.
    """
    case = _small()
    mixed = {"pupil": "interpixel", "focus": "pixel"}
    x = pupil_coords(case, mixed)
    u = focus_coords(case, mixed)
    field = circular_aperture(case, mixed).astype(np.complex128)
    k = np.exp(-2j * np.pi * np.outer(u, x))
    out = (k @ field) @ k.T * case.dx_pupil**2

    good = validate.compare(out, reference_field(case, mixed), case)
    assert good.gate == "pass" and good.rel_l2 < 1e-13

    bad = validate.compare(out, reference_field(case, "pixel"), case)
    assert bad.gate == "fail"
    assert bad.peak_offset_px == (0, 0), "the peak does not move; only the phase does"
    assert 1e-4 < bad.rel_l2 < 1e-1
