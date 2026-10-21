"""Propagation references must model the aperture declared by the case."""
from dataclasses import replace

import numpy as np
import pytest

from dragrace import adapter as adapters
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import aperture_mask, focus_coords, pupil_coords
from dragrace.reference import reference_field
from dragrace.validate import compare
from dragrace.worker import _measure


def _case(algorithm="matrix_dft", obstructed=True):
    free_space = algorithm in ("angular_spectrum", "fresnel_tf")
    pupil = {"diameter_m": 0.01, "samples_across_diameter": 32,
             "array_samples": 64 if free_space else 32}
    if obstructed:
        pupil.update(aperture="obstructed", obstruction={
            "secondary_ratio": 0.3, "spider_width_ratio": 0.02,
            "spider_angles_deg": [-30, 30], "spider_span": "radius",
        })
    spec = {
        "id": "obstructed", "kind": "plane_to_plane" if free_space else "pupil_to_focus",
        "algorithm_class": algorithm, "wavelength_m": 500e-9, "pupil": pupil,
        "accuracy": {"reference": f"internal_{algorithm}" if free_space else "internal_mft",
                     "max_rel_l2": 1e-10},
    }
    if free_space:
        spec["propagation"] = {"distance_m": 1.0}
    else:
        spec["output"] = {"focal_length_m": 1.0, "samples_per_lambda_f_d": 4.0,
                          "extent_lambda_f_d": 8.0}
    return Case.from_dict(spec)


@pytest.mark.parametrize("algorithm", ["matrix_dft", "fft", "angular_spectrum", "fresnel_tf"])
def test_obstructed_baseline_passes_its_reference(algorithm):
    case = _case(algorithm)
    config = Config.from_yaml("configs/cpu_numpy_1t.yaml")
    adapters.discover("adapters")
    ad = adapters.get("numpy_baseline")
    assert ad.configure(config)
    state = ad.build(case, config)
    try:
        result = compare(ad.to_host(ad.propagate(state)), reference_field(case), case)
        assert result.gate == "pass", result
    finally:
        ad.teardown(state)


@pytest.mark.parametrize("centering", [
    "pixel", "interpixel", {"pupil": "interpixel", "focus": "pixel"},
])
def test_obstructed_mft_matches_direct_sum(centering):
    case = _case()
    x = pupil_coords(case, centering)
    u = focus_coords(case, centering)
    pupil = aperture_mask(case, centering)
    # Check an independent scalar diffraction sum at on- and off-axis samples.
    actual = reference_field(case, centering)
    for row, col in [(16, 16), (13, 18), (7, 9)]:
        phase = np.exp(-2j * np.pi * (u[row] * x[:, None] + u[col] * x[None, :]))
        expected = np.sum(pupil * phase) * case.dx_pupil ** 2
        np.testing.assert_allclose(actual[row, col], expected, rtol=1e-12, atol=1e-14)


def test_obstructed_case_rejects_clear_aperture_airy_reference():
    case = _case()
    with pytest.raises(ValueError, match="analytic_airy.*circular"):
        replace(case, accuracy=replace(case.accuracy, reference="analytic_airy")).validate()


@pytest.mark.parametrize("algorithm, obstructed, has_airy_check", [
    ("matrix_dft", False, True),
    ("matrix_dft", True, False),
    ("angular_spectrum", False, False),
    ("fresnel_tf", False, False),
])
def test_worker_only_adds_airy_diagnostic_for_a_clear_focal_pupil(
        algorithm, obstructed, has_airy_check, tmp_path):
    case = _case(algorithm, obstructed)
    config = Config.from_yaml("configs/cpu_numpy_1t.yaml")
    adapters.discover("adapters")
    ad = adapters.get("numpy_baseline")
    assert ad.configure(config)
    result = _measure(ad, case, config, "ledger", tmp_path, "numpy_baseline")
    assert result["status"] == "ok", result
    assert result["accuracy"]["gate"] == "pass"
    assert ("physics_check_analytic_airy" in result["accuracy"]) is has_airy_check
