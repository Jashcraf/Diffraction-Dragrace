"""The FLOP model is checked against hand-derivations.

If ideal_work() is wrong, every efficiency number downstream is wrong in the
same direction and nothing else in the suite would catch it. The values below
are worked out by hand in docs/flop_model.md.
"""
import math

import pytest

from dragrace.case import Case
from dragrace.flops import fft_2d, ideal_work, zgemm
from dragrace.flops.model import gradient_ideal_work


def test_zgemm_convention():
    # zgemm(M,K,N) = 8*M*K*N: a complex FMA is 6 (mul) + 2 (add) real flops.
    assert zgemm(2, 3, 4) == 8 * 2 * 3 * 4


def test_fft_2d_counts_both_passes_once():
    # 5*N1*N2*log2(N1*N2) == 10*N^2*log2(N) for a square transform. Getting this
    # wrong by a factor of 2 was a real bug during development.
    n = 4096
    assert fft_2d(n) == pytest.approx(10 * n * n * math.log2(n))
    assert fft_2d(n) == pytest.approx(2.013e9, rel=1e-3)


def test_mft_case_matches_hand_derivation():
    case = Case.from_yaml("cases/pupil_to_focus/mft_n1024_q4.yaml")
    w = ideal_work(case)
    # 8 * N_p * N_f * (N_p + N_f) = 8 * 1024 * 128 * 1152
    assert w.flops == pytest.approx(8 * 1024 * 128 * (1024 + 128))
    assert w.flops == pytest.approx(1.208e9, rel=1e-3)
    # Kernels are precomputed in this case, so no transcendentals are charged.
    assert w.tops == 0.0


def test_fft_case_matches_hand_derivation():
    case = Case.from_yaml("cases/pupil_to_focus/fft_n1024_q4.yaml")
    assert case.n_fft == 4096                       # q=4 x N_D=1024
    assert ideal_work(case).flops == pytest.approx(2.013e9, rel=1e-3)


def test_fft_costs_more_than_mft_at_this_field_of_view():
    """The pairing that motivates algorithm_class being part of the case.

    At W = 32 lambda/D the MFT is cheaper; at a wide enough field the ranking
    inverts, because the FFT's cost does not grow with the output extent while
    the MFT's does.
    """
    mft = ideal_work(Case.from_yaml("cases/pupil_to_focus/mft_n1024_q4.yaml"))
    fft = ideal_work(Case.from_yaml("cases/pupil_to_focus/fft_n1024_q4.yaml"))
    assert mft.flops < fft.flops
    assert fft.flops / mft.flops == pytest.approx(1.667, rel=0.01)


def test_gradient_floor_is_twice_forward_and_p_independent():
    case = Case.from_yaml("cases/gradient/grad_zernike_p15_n256.yaml")
    g = gradient_ideal_work(case)
    fwd = ideal_work(case)
    # Reverse mode over a linear propagation: backward ~= forward. The basis
    # terms add on top, and are the only P-dependent part.
    assert g.flops > 2 * fwd.flops
    assert g.flops < 2.5 * fwd.flops


def test_arithmetic_intensity_is_finite_and_positive():
    for f in ("cases/pupil_to_focus/mft_n1024_q4.yaml",
              "cases/pupil_to_focus/fft_n1024_q4.yaml"):
        w = ideal_work(Case.from_yaml(f))
        assert 0 < w.arithmetic_intensity < 1e6
