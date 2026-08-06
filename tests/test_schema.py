"""Every case and config in the repo must load and validate."""
from pathlib import Path

import pytest

from dragrace.case import Case
from dragrace.config import Config

CASES = [p for p in Path("cases").rglob("*.yaml") if not p.stem.startswith("sweep_")]
CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_loads(path):
    case = Case.from_yaml(path)
    assert case.id == path.stem
    assert case.n_focus > 0 and case.n_pupil > 0
    assert case.n_focus % 2 == 0, "focal grids are forced even so the centre is at N//2"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    cfg = Config.from_yaml(path)
    assert cfg.id == path.stem
    assert cfg.threads >= 1


def test_case_rejects_analytic_airy_with_aberration():
    """Guard rail: the closed form is only valid for an unaberrated pupil."""
    base = Case.from_yaml("cases/pupil_to_focus/mft_n1024_q4.yaml")
    d = {
        "id": "bad", "kind": "pupil_to_focus", "algorithm_class": "matrix_dft",
        "wavelength_m": base.wavelength_m,
        "pupil": {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 64,
                  "aberration": {"coefficients": {4: 0.1}}},
        "output": {"focal_length_m": 1.0, "samples_per_lambda_f_d": 4.0,
                   "extent_lambda_f_d": 8.0},
        "accuracy": {"reference": "analytic_airy", "max_rel_l2": 1e-6},
    }
    with pytest.raises(ValueError, match="analytic_airy"):
        Case.from_dict(d)


def test_case_rejects_unreachable_precision_gate():
    """complex64 cannot meet a float64-grade tolerance; fail at load, not at run."""
    d = {
        "id": "bad2", "kind": "pupil_to_focus", "algorithm_class": "matrix_dft",
        "wavelength_m": 500e-9, "dtype": "complex64",
        "pupil": {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 64},
        "output": {"focal_length_m": 1.0, "samples_per_lambda_f_d": 4.0,
                   "extent_lambda_f_d": 8.0},
        "accuracy": {"reference": "internal_mft", "max_rel_l2": 1e-10},
    }
    with pytest.raises(ValueError, match="complex64"):
        Case.from_dict(d)


def test_every_config_names_an_environment():
    """A config with no conda_env would run in whatever interpreter happens to
    be active, which makes its label meaningless."""
    for path in CONFIGS:
        cfg = Config.from_yaml(path)
        assert cfg.conda_env, f"{cfg.id} does not name a conda environment"
