"""Correctness gates for the reference implementation.

These are the tests CI runs. They validate physics and the adjoint chain; they
say nothing about speed, and CI must never be used for timing (shared runners
are far too noisy).
"""
import numpy as np
import pytest

from dragrace.adapter import discover, get
from dragrace.case import Case
from dragrace.config import Config
from dragrace.grid import gradient_parameters, zernike_basis, circular_aperture
from dragrace.reference import airy_field, loss_and_reference_gradient, reference_field
from dragrace.validate import compare, compare_gradients

CFG = Config.from_yaml("configs/cpu_numpy_1t.yaml")


@pytest.fixture(scope="module", autouse=True)
def _adapters():
    discover("adapters")


def _run(case_file):
    case = Case.from_yaml(case_file)
    ad = get("numpy_baseline")
    assert ad.configure(CFG) is True
    state = ad.build(case, CFG)
    return case, ad.to_host(ad.propagate(state))


@pytest.mark.parametrize("case_file", [
    "cases/pupil_to_focus/mft_n1024_q4.yaml",
    "cases/pupil_to_focus/fft_n1024_q4.yaml",
])
def test_accuracy_gate(case_file):
    case, out = _run(case_file)
    assert out.shape == (case.n_focus, case.n_focus)
    assert out.dtype == np.dtype(case.dtype)
    c = compare(out, reference_field(case), case)
    assert c.gate == "pass", c


def test_fft_and_mft_agree_to_roundoff():
    """The genuinely independent cross-check.

    For a matrix_dft case the internal reference performs the same operations as
    the baseline adapter, so agreement there is near-trivial. The FFT path
    reaches the same focal samples by a completely different route, so this is
    the comparison that actually constrains both.
    """
    _, mft = _run("cases/pupil_to_focus/mft_n1024_q4.yaml")
    case, fft = _run("cases/pupil_to_focus/fft_n1024_q4.yaml")
    rel = np.linalg.norm(fft - mft) / np.linalg.norm(mft)
    assert rel < 1e-12, f"FFT and MFT disagree at {rel:.3e}"


def test_analytic_airy_physics_check():
    """Ungated but bounded: the pixelated aperture differs from the continuous
    one at ~1e-3, and that number should not drift."""
    case, out = _run("cases/pupil_to_focus/mft_n1024_q4.yaml")
    c = compare(out, airy_field(case), case)
    assert 1e-5 < c.rel_l2 < 5e-3
    assert c.peak_ratio == pytest.approx(1.0, abs=1e-3)


def test_zernike_basis_is_unit_rms_over_the_aperture():
    case = Case.from_yaml("cases/gradient/grad_zernike_p15_n256.yaml")
    basis = zernike_basis(case, [4, 5, 6, 11])
    mask = circular_aperture(case) > 0
    for mode in basis:
        assert np.sqrt(np.mean(mode[mask] ** 2)) == pytest.approx(1.0, rel=1e-12)


def test_gradient_matches_finite_differences():
    """The gradient board's correctness gate, run against the baseline's
    hand-written adjoint."""
    case = Case.from_yaml("cases/gradient/grad_zernike_p15_n256.yaml")
    ad = get("numpy_baseline")
    ad.configure(CFG)
    state = ad.build_gradient(case, CFG)
    loss, grad = ad.gradient(state)

    _, theta0, _ = gradient_parameters(case)
    ref_loss, ref_grad = loss_and_reference_gradient(case, theta0)

    assert loss == pytest.approx(ref_loss, rel=1e-12)
    c = compare_gradients(np.asarray(grad), ref_grad)
    assert c.gate == "pass", c
    # A scale ratio of 2 or -1 is the classic Wirtinger convention slip; the
    # cosine check catches it even when per-component tolerance would not.
    assert c.scale_ratio == pytest.approx(1.0, rel=1e-6)


def test_gradient_is_parameter_count_independent_in_shape():
    case = Case.from_yaml("cases/gradient/grad_zernike_p15_n256.yaml")
    ad = get("numpy_baseline")
    ad.configure(CFG)
    _, grad = ad.gradient(ad.build_gradient(case, CFG))
    assert np.asarray(grad).shape == (case.parameters.count,)
    assert np.isrealobj(np.asarray(grad))
