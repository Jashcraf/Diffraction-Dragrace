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
    quantity: str = "field"  # field | intensity | transmission -- what was gated
    #: What `rel_l2` actually holds. Every propagation board fits a scale and
    #: reports a relative L2 residual; the aperture board cannot, because the
    #: codes disagree about antialiasing and an L2 would then gate on edge
    #: treatment rather than on geometry. Naming the metric in the result keeps
    #: the two from being read as the same number.
    metric: str = "relative_l2"
    #: Aperture board only. IoU of the binarised masks (the gated quantity
    #: there), the open-area fractions either side, and the raw L2 that IoU
    #: replaced -- reported so the edge disagreement stays visible.
    iou: float | None = None
    fill_fraction: float | None = None
    fill_fraction_reference: float | None = None
    edge_relative_l2: float | None = None
    #: Phase-retrieval board only. `rel_l2` there is the coefficient error over
    #: the OBSERVABLE modes; these carry the same error over all P (piston
    #: included, which no PSF can constrain), the worst single mode in waves,
    #: and the distance to the twin solution -- so a run that converged to the
    #: sign-flipped wavefront is diagnosable from the result file alone instead
    #: of looking like a generic accuracy failure.
    coefficient_rel_l2_all: float | None = None
    max_coefficient_error_waves: float | None = None
    twin_rel_l2: float | None = None

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

    return compare_psf(np.abs(test.astype(np.complex128)) ** 2, np.abs(ref) ** 2,
                       case, case.accuracy.max_rel_l2, case.accuracy.reference)


def compare_psf(test: np.ndarray, ref: np.ndarray, case: Case,
                max_rel_l2: float, reference: str,
                quantity: str = "intensity") -> Comparison:
    """Compare two real intensity PSFs, fitting out one overall scale.

    The scale has to be fitted because these codes disagree about PSF
    normalisation by design -- POPPY divides by the entrance flux, prysm carries
    dx^2, HCIPy does neither -- and none of those is wrong. What is left after
    one scale factor is a real disagreement about the optics.

    Split out from compare_intensity so the phase-retrieval board can reuse it
    with its own tolerance: there this compares each code's forward model
    against the harness reference at the truth coefficients, which is a
    different question with a different acceptable error from a propagation
    board's gate.
    """
    t = np.asarray(test, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError(
            f"shape mismatch: adapter returned {t.shape}, reference is {r.shape}. "
            f"Case {case.id!r} specifies N_f={case.n_focus} on the canonical focal grid."
        )

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
        gate="pass" if rel <= max_rel_l2 else "fail",
        reference=reference,
        quantity=quantity,
    )


def compare_aperture(test: np.ndarray, ref: np.ndarray, case: Case) -> Comparison:
    """Gate a *drawn* pupil on geometry rather than on pixel values.

    The aperture board is the one place where an L2 residual is the wrong gate.
    These codes disagree about antialiasing by design -- HCIPy returns a
    two-valued mask, prysm and lentil antialias their edges by default, POPPY
    does its own thing -- and the ELT has 798 segments, so edge pixels are a
    large fraction of the pupil at every size this scan reaches (~6% at
    N=2048, more below it). An L2 gate would therefore reject codes for their
    edge treatment, which is a modelling choice each is entitled to, while
    saying almost nothing about whether they drew the right telescope.

    So the gated number is `1 - IoU` of the masks binarised at half
    transmission. Binarising splits an antialiased edge pixel the same way for
    everyone, which makes IoU nearly blind to the thing that is legitimately
    different and sharp about the things that are not: a missing central
    obscuration, absent spiders, or segments the wrong size all move it by far
    more than antialiasing does.

    The raw L2 is not discarded -- it comes back as `edge_relative_l2`, which is
    a decent proxy for how much antialiasing a code applies, and the two fill
    fractions are reported alongside so "drew the right shape, wrong size" is
    visible directly.
    """
    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if test.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: adapter drew {test.shape}, reference is {ref.shape}. "
            f"Case {case.id!r} specifies a {case.n_pupil}x{case.n_pupil} pupil. An "
            f"adapter whose aperture routine sizes its own output must resample or "
            f"pad to the case grid in build(), not return its native size."
        )

    a = test > 0.5
    b = ref > 0.5
    union = int((a | b).sum())
    iou = float((a & b).sum() / union) if union else 0.0

    denom = float(np.linalg.norm(ref))
    edge_l2 = float(np.linalg.norm(test - ref) / denom) if denom else float("inf")

    # Centroid offset, in pixels: a whole-pupil shift is a centring convention
    # difference and shows up here rather than as a mysteriously poor IoU.
    def _centroid(m):
        tot = m.sum()
        if tot == 0:
            return (0.0, 0.0)
        yy, xx = np.indices(m.shape)
        return (float((yy * m).sum() / tot), float((xx * m).sum() / tot))

    cy_t, cx_t = _centroid(test)
    cy_r, cx_r = _centroid(ref)

    rel = 1.0 - iou
    return Comparison(
        rel_l2=rel,
        scale_abs=1.0,
        scale_phase_rad=None,
        conjugated=None,
        peak_ratio=float(test.max() / ref.max()) if ref.max() else float("inf"),
        peak_offset_px=(int(round(cy_t - cy_r)), int(round(cx_t - cx_r))),
        gate="pass" if rel <= case.accuracy.max_rel_l2 else "fail",
        reference=case.accuracy.reference,
        quantity="transmission",
        metric="one_minus_iou",
        iou=iou,
        fill_fraction=float(test.mean()),
        fill_fraction_reference=float(ref.mean()),
        edge_relative_l2=edge_l2,
    )


def compare_retrieval(test: np.ndarray, ref: np.ndarray, case: Case) -> Comparison:
    """Gate a recovered Zernike coefficient vector against the truth.

    The gated number is the relative L2 error over the OBSERVABLE modes. Piston
    is excluded from it because a PSF cannot see piston at all -- dL/dtheta_1 is
    identically zero, measured at 7.5e-20 through reverse-mode AD -- so the
    optimiser leaves it wherever it started and including it would charge every
    code for a mode none of them could possibly recover. It is not swept away:
    `coefficient_rel_l2_all` reports the error with piston in.

    The distance to the twin solution is reported alongside, because that is the
    one failure this board is genuinely exposed to and it needs to be
    distinguishable from ordinary inaccuracy. A pupil that is centro-symmetric
    makes OPD phi(x) and -phi(-x) give the identical PSF; a code that converged
    there scores rel_l2 ~ 1.27 and twin_rel_l2 ~ 0, which says "found the other
    valid answer" rather than "computed something wrong".
    """
    from .retrieval import observable_slice, twin_coefficients

    test = np.asarray(test, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if test.shape != ref.shape:
        raise ValueError(
            f"coefficient shape mismatch: adapter returned {test.shape}, the case "
            f"specifies {ref.shape} (P={case.retrieval.count} Noll modes starting "
            f"at {case.retrieval.first_noll})."
        )

    def _rel(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(b))
        return float(np.linalg.norm(a - b) / denom) if denom else float("inf")

    sl = observable_slice(case)
    rel = _rel(test[sl], ref[sl])
    twin = twin_coefficients(case, ref)

    norm_t, norm_r = float(np.linalg.norm(test[sl])), float(np.linalg.norm(ref[sl]))
    return Comparison(
        rel_l2=rel,
        # No normalisation is fitted out here, unlike a field comparison: a
        # coefficient vector is already in physical units (waves RMS), so a
        # scale factor would be a real error rather than a convention.
        scale_abs=(norm_t / norm_r) if norm_r else float("inf"),
        scale_phase_rad=None,
        conjugated=None,
        peak_ratio=(norm_t / norm_r) if norm_r else float("inf"),
        peak_offset_px=(0, 0),
        gate="pass" if rel <= case.accuracy.max_rel_l2 else "fail",
        reference=case.accuracy.reference,
        quantity="zernike_coefficients",
        metric="coefficient_relative_l2",
        coefficient_rel_l2_all=_rel(test, ref),
        max_coefficient_error_waves=float(np.max(np.abs(test[sl] - ref[sl])))
        if test[sl].size else 0.0,
        twin_rel_l2=_rel(test[sl], twin[sl]),
    )


def gate_message(c: Comparison, case: Case) -> str:
    if c.metric == "coefficient_relative_l2":
        head = (f"retrieval {'pass' if c.gate == 'pass' else 'FAIL'}: coefficient "
                f"rel_l2={c.rel_l2:.3e} over the observable modes (vs "
                f"{case.accuracy.max_rel_l2:.1e})")
        if c.gate == "pass":
            return head
        detail = (f". Worst mode off by {c.max_coefficient_error_waves:.3e} waves; "
                  f"with piston included rel_l2={c.coefficient_rel_l2_all:.3e}")
        if c.twin_rel_l2 is not None and c.twin_rel_l2 < c.rel_l2:
            detail += (f". This is the TWIN SOLUTION -- distance to it is "
                       f"{c.twin_rel_l2:.3e}, closer than to the truth. The optimiser "
                       f"found a wavefront that reproduces the PSF equally well, "
                       f"which means this case's pupil is centro-symmetric enough "
                       f"for the two to be indistinguishable. Not an accuracy bug: "
                       f"see src/dragrace/retrieval.py on spider_span")
        return head + detail

    if c.metric == "one_minus_iou":
        head = (f"geometry {'pass' if c.gate == 'pass' else 'FAIL'}: "
                f"IoU={c.iou:.4f} (1-IoU={c.rel_l2:.3e} vs "
                f"{case.accuracy.max_rel_l2:.1e})")
        detail = (f"  fill={c.fill_fraction:.4f} against {c.fill_fraction_reference:.4f}, "
                  f"antialiasing residual={c.edge_relative_l2:.3f}")
        if c.gate == "pass":
            return head
        if c.peak_offset_px != (0, 0):
            detail += (f", centroid offset {c.peak_offset_px} px -- a whole-pupil "
                       f"shift is a centring convention, not a layout error")
        return head + "." + detail
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
