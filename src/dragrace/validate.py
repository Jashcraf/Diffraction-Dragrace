"""Accuracy gates.

Cross-code field comparison has to tolerate two legitimate differences and
tolerate nothing else:

  normalisation   codes disagree on whether the PSF sums to the pupil energy,
                  peaks at 1, or carries dx^2. A single complex scale factor
                  absorbs all of these.
  phase sign      exp(+ikz) versus exp(-ikz) is a convention, not an error.
                  Conjugating the test field absorbs it.

Everything left after fitting those two is a real disagreement. Both the fitted
scale and the conjugation flag are *reported* rather than silently discarded --
a code whose normalisation is off by 4x is not wrong, but it is worth knowing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .case import Case


@dataclass
class Comparison:
    rel_l2: float            # shape error after fitting scale; the gated number
    scale_abs: float         # |alpha|: normalisation relative to the reference
    scale_phase_rad: float | None   # arg(alpha): global piston, physically
                             # irrelevant. None when only intensity was compared.
    conjugated: bool | None  # whether the test field matched the conjugate.
                             # None when only intensity was compared -- squaring
                             # the modulus destroys the sign of the phase, so the
                             # question has no answer rather than a False answer.
    peak_ratio: float
    peak_offset_px: tuple[int, int]
    gate: str                # pass | fail
    reference: str
    quantity: str = "field"  # field | intensity -- what was actually gated

    def to_dict(self) -> dict:
        return asdict(self)


def _fit(test: np.ndarray, ref: np.ndarray) -> tuple[float, complex]:
    """Least-squares complex scale minimising ||test - alpha*ref||, and the
    resulting relative residual."""
    denom = np.vdot(ref, ref)
    if denom == 0:
        return float("inf"), 0j
    alpha = np.vdot(ref, test) / denom
    resid = test - alpha * ref
    scale = np.linalg.norm(alpha * ref)
    if scale == 0:
        return float("inf"), alpha
    return float(np.linalg.norm(resid) / scale), complex(alpha)


def compare(test: np.ndarray, ref: np.ndarray, case: Case) -> Comparison:
    test = np.asarray(test)
    ref = np.asarray(ref)
    if test.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: adapter returned {test.shape}, reference is {ref.shape}. "
            f"Case {case.id!r} specifies N_f={case.n_focus} on the canonical focal grid."
        )

    direct, alpha_d = _fit(test.astype(np.complex128), ref)
    conj, alpha_c = _fit(np.conj(test).astype(np.complex128), ref)
    conjugated = conj < direct
    rel, alpha = (conj, alpha_c) if conjugated else (direct, alpha_d)

    t_int = np.abs(test) ** 2
    r_int = np.abs(ref) ** 2
    ti = np.unravel_index(int(np.argmax(t_int)), t_int.shape)
    ri = np.unravel_index(int(np.argmax(r_int)), r_int.shape)
    peak_ratio = float(t_int[ti] / r_int[ri]) if r_int[ri] > 0 else float("inf")

    return Comparison(
        rel_l2=rel,
        scale_abs=float(abs(alpha)),
        scale_phase_rad=float(np.angle(alpha)),
        conjugated=bool(conjugated),
        peak_ratio=peak_ratio,
        peak_offset_px=(int(ti[0] - ri[0]), int(ti[1] - ri[1])),
        gate="pass" if rel <= case.accuracy.max_rel_l2 else "fail",
        reference=case.accuracy.reference,
    )


def compare_intensity(test: np.ndarray, ref: np.ndarray, case: Case) -> Comparison:
    """Gate |E|^2 rather than E, for codes whose documented output is a PSF.

    PROPER's `prop_end` returns intensity unless asked for the field, and its
    focal amplitude matches the reference to ~1e-7 while its *phase* carries a
    residual quadratic curvature -- it propagates through a lens and tracks a
    reference sphere rather than assuming the Fraunhofer limit. A single complex
    scale cannot absorb a quadratic, so gating the field would reject a PSF that
    is correct to a part in ten million.

    What is given up is stated rather than hidden: squaring the modulus destroys
    the phase, so `conjugated` and `scale_phase_rad` come back None and an
    intensity-gated row must not be used for any phase-sensitive claim. That is
    why this is a per-adapter declaration and not a fallback the harness reaches
    for when the field comparison fails.
    """
    test = np.asarray(test)
    ref = np.asarray(ref)
    if test.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: adapter returned {test.shape}, reference is {ref.shape}. "
            f"Case {case.id!r} specifies N_f={case.n_focus} on the canonical focal grid."
        )

    t = np.abs(test.astype(np.complex128)) ** 2
    r = np.abs(ref) ** 2
    denom = float((r * r).sum())
    a = float((t * r).sum() / denom) if denom else float("inf")
    scale = np.linalg.norm(a * r)
    rel = float(np.linalg.norm(t - a * r) / scale) if scale else float("inf")

    ti = np.unravel_index(int(np.argmax(t)), t.shape)
    ri = np.unravel_index(int(np.argmax(r)), r.shape)
    peak_ratio = float(t[ti] / r[ri]) if r[ri] > 0 else float("inf")

    return Comparison(
        rel_l2=rel,
        # sqrt so the number means the same thing as the field path's scale_abs:
        # the amplitude ratio, not the intensity ratio.
        scale_abs=float(np.sqrt(a)) if a == a and a >= 0 else float("inf"),
        scale_phase_rad=None,
        conjugated=None,
        peak_ratio=peak_ratio,
        peak_offset_px=(int(ti[0] - ri[0]), int(ti[1] - ri[1])),
        gate="pass" if rel <= case.accuracy.max_rel_l2 else "fail",
        reference=case.accuracy.reference,
        quantity="intensity",
    )


def gate_message(c: Comparison, case: Case) -> str:
    if c.gate == "pass":
        return f"accuracy pass: rel_l2={c.rel_l2:.3e} <= {case.accuracy.max_rel_l2:.1e}"
    extra = ""
    if c.peak_offset_px != (0, 0):
        extra = (f"  PSF peak is offset by {c.peak_offset_px} px -- likely a grid-centring "
                 f"convention mismatch rather than a propagation error.")
    return (f"accuracy FAIL: rel_l2={c.rel_l2:.3e} > {case.accuracy.max_rel_l2:.1e} "
            f"on {c.quantity} (scale={c.scale_abs:.6g}, "
            f"conjugated={c.conjugated}).{extra}")


# --------------------------------------------------------- gradient board ---
@dataclass
class GradientComparison:
    max_rel_err: float
    cosine_similarity: float
    scale_ratio: float       # median g_test/g_ref: catches a uniform factor of 2
    gate: str

    def to_dict(self) -> dict:
        return asdict(self)


def compare_gradients(test: np.ndarray, ref: np.ndarray,
                      rtol: float = 1e-6, cos_tol: float = 1e-9) -> GradientComparison:
    """Compare two real parameter-space gradients.

    Both codes return d(real loss)/d(real theta), so there is no Wirtinger
    ambiguity to reconcile here -- that ambiguity lives in the intermediate
    complex cotangents, which is exactly why the board is defined at the
    parameter level and never compares intermediates.

    The cosine check is not redundant with the per-component one: a uniform
    factor of 2 (the classic Wirtinger slip) can hide under a loose relative
    tolerance while showing up immediately as a scale ratio.
    """
    test = np.asarray(test, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if test.shape != ref.shape:
        raise ValueError(f"gradient shape mismatch: {test.shape} vs {ref.shape}")

    denom = np.where(np.abs(ref) > 0, np.abs(ref), np.nan)
    max_rel = float(np.nanmax(np.abs(test - ref) / denom))
    nt, nr = np.linalg.norm(test), np.linalg.norm(ref)
    cos = float(np.dot(test, ref) / (nt * nr)) if nt > 0 and nr > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.nanmedian(np.where(np.abs(ref) > 0, test / ref, np.nan)))

    ok = (max_rel <= rtol) and (1.0 - cos <= cos_tol)
    return GradientComparison(max_rel, cos, ratio, "pass" if ok else "fail")
