"""A zero reference derivative must not hide a spurious parameter sensitivity."""
import numpy as np
import pytest

from dragrace.validate import compare_gradients


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_spurious_derivative_at_zero_reference_fails(scale):
    result = compare_gradients(scale * np.array([1.0, 1e-5]),
                               scale * np.array([1.0, 0.0]))
    # The direction is close enough to pass the cosine gate, but the extra
    # derivative exceeds the component tolerance relative to the vector scale.
    assert 1.0 - result.cosine_similarity < 1e-9
    assert result.max_rel_err == pytest.approx(1e-5)
    assert result.gate == "fail"


def test_zero_reference_component_allows_roundoff_at_the_gradient_scale():
    result = compare_gradients(np.array([1.0, 1e-13]), np.array([1.0, 0.0]))
    assert result.gate == "pass"
    assert result.max_rel_err == pytest.approx(1e-13)
    assert compare_gradients(np.array([1.0, 1e-13]), np.array([1.0, 0.0]),
                             rtol=1e-14).gate == "fail"


def test_identical_zero_gradients_pass_without_warnings():
    result = compare_gradients(np.zeros(3), np.zeros(3))
    assert result.gate == "pass"
    assert result.max_rel_err == 0.0


def test_nonzero_gradient_fails_against_an_all_zero_reference():
    assert compare_gradients(np.array([0.0, 1e-12]), np.zeros(2)).gate == "fail"


@pytest.mark.parametrize("test", [[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
def test_nonzero_component_scale_and_sign_are_still_gated(test):
    result = compare_gradients(np.array(test), np.array([1.0, 0.0]))
    assert result.gate == ("pass" if test[0] == 1.0 else "fail")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["test", "reference"])
def test_nonfinite_gradients_fail_without_warnings(invalid, side):
    test, ref = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    (test if side == "test" else ref)[1] = invalid
    assert compare_gradients(test, ref).gate == "fail"


def test_empty_gradients_are_rejected():
    with pytest.raises(ValueError, match="empty gradients"):
        compare_gradients(np.array([]), np.array([]))


def test_worker_rejects_spurious_sensitivity_before_timing(monkeypatch, tmp_path):
    from dragrace import adapter, metrics, reference
    from dragrace.case import Case
    from dragrace.config import Config
    from dragrace.worker import _measure

    case = Case.from_yaml("cases/gradient/grad_zernike_p15_n256.yaml")
    config = Config.from_yaml("configs/cpu_numpy_1t.yaml")
    adapter.discover("adapters")
    ad = adapter.get("numpy_baseline")
    assert ad.configure(config)
    expected = np.zeros(case.parameters.count)
    expected[0] = 1.0
    actual = expected.copy()
    actual[1] = 1e-5
    monkeypatch.setattr(ad, "gradient", lambda state: (1.0, actual))
    monkeypatch.setattr(reference, "loss_and_reference_gradient",
                        lambda *args: (1.0, expected))
    monkeypatch.setattr(metrics, "time_gradient",
                        lambda *args: pytest.fail("timed an invalid gradient"))

    result = _measure(ad, case, config, "gradient", tmp_path, "numpy_baseline")
    assert result["status"] == "accuracy_fail"
    assert result["gradient_accuracy"]["gate"] == "fail"
    assert "timing" not in result
