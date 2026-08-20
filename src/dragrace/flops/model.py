"""Analytic cost model: the "physics floor".

Every number this repo publishes about algorithmic efficiency depends on the
conventions below, so they are stated explicitly and are open to dispute. See
docs/flop_model.md for the derivations, and tests/test_ideal_flops.py for the
hand-checked values.

Two currencies, kept separate on purpose:

  flops   real floating-point operations, FMA-class.
  tops    transcendental evaluations (cexp, sin, cos). Their cost relative to a
          FLOP ranges over roughly 10-40x depending on whether the build gets
          SVML/libmvec vectorisation, so folding them in with a fudge factor
          would silently encode a machine assumption into a supposedly
          hardware-independent metric.

Plus `bytes`, for the memory-bound operations (padding, fftshift, elementwise),
which is what makes the roofline classification possible.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..case import Case

# ------------------------------------------------------------- conventions --
COMPLEX_ADD = 2      # 2 real adds
COMPLEX_MUL = 6      # 4 real mults + 2 real adds
COMPLEX_FMA = 8      # complex mul + complex add
ZGEMM_PER_MAC = 8    # zgemm(M,K,N) = 8*M*K*N real flops
FFT_RADIX2 = 5       # canonical 5*N*log2(N) real flops for a length-N C2C FFT


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass
class Work:
    """An estimate of the work a computation requires."""

    flops: float = 0.0
    tops: float = 0.0
    bytes: float = 0.0
    detail: str = ""

    def __add__(self, other: "Work") -> "Work":
        return Work(self.flops + other.flops, self.tops + other.tops,
                    self.bytes + other.bytes, "; ".join(filter(None, [self.detail, other.detail])))

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arithmetic_intensity"] = self.arithmetic_intensity
        return d


# -------------------------------------------------------------- primitives --
def fft_1d(n: int) -> float:
    return FFT_RADIX2 * n * math.log2(n) if n > 1 else 0.0


def fft_2d(n1: int, n2: int | None = None) -> float:
    """Row-column 2-D C2C FFT: 5*N1*N2*log2(N1*N2)."""
    n2 = n1 if n2 is None else n2
    return FFT_RADIX2 * n1 * n2 * math.log2(n1 * n2)


def zgemm(m: int, k: int, n: int) -> float:
    return ZGEMM_PER_MAC * m * k * n


def elementwise_mul(n: int) -> float:
    return COMPLEX_MUL * n


def itemsize(dtype: str) -> int:
    return 8 if dtype == "complex64" else 16


# ------------------------------------------------------- the physics floor --
def ideal_work(case: Case) -> Work:
    """Cost of an idealised implementation of this case's physics.

    Derived from the case alone -- no library involved. This is the denominator
    of the algorithmic-overhead metric A = flops_actual / flops_ideal.
    """
    cls = case.algorithm_class
    n_p, n_f, n_d = case.n_pupil, case.n_focus, case.n_across
    isz = itemsize(case.dtype)

    if cls in ("matrix_dft", "czt") and cls == "matrix_dft":
        # Two GEMMs: (N_f x N_p)(N_p x N_p) then (N_f x N_p)(N_p x N_f).
        flops = zgemm(n_f, n_p, n_p) + zgemm(n_f, n_p, n_f)
        # Kernel matrices are 2 * N_p * N_f complex exponentials -- charged only
        # when the case says they are rebuilt per call rather than hoisted into
        # build(). Which of those a library does is a real design difference and
        # one of the things this suite is meant to expose.
        tops = 2 * n_p * n_f if case.basis_caching == "per_call" else 0.0
        nbytes = isz * (n_p * n_p + 2 * n_p * n_f + n_f * n_f)
        return Work(flops, tops, nbytes, f"2 zgemm: ({n_f}x{n_p}x{n_p}) + ({n_f}x{n_p}x{n_f})")

    if cls == "fft":
        n = case.n_fft
        # fft_2d already covers both row and column passes: 5*N^2*log2(N^2)
        # == 10*N^2*log2(N). Multiplying by 2 here would double-count.
        flops = fft_2d(n)
        nbytes = isz * 3 * n * n         # pad + transform + shift traffic
        return Work(flops, 0.0, nbytes, f"1 fft2({n}x{n}) with padding factor q={case.q:g}")

    if cls in ("fresnel_tf", "angular_spectrum"):
        n = case.n_fft
        flops = 2 * fft_2d(n) + elementwise_mul(n * n)
        tops = n * n if case.basis_caching == "per_call" else 0.0
        nbytes = isz * 5 * n * n
        return Work(flops, tops, nbytes, f"2 fft2({n}x{n}) + transfer-function multiply")

    if cls == "fresnel_ir":
        n = case.n_fft
        flops = fft_2d(n) + 2 * elementwise_mul(n * n)
        tops = 2 * n * n if case.basis_caching == "per_call" else 0.0
        return Work(flops, tops, isz * 4 * n * n, f"1 fft2({n}x{n}) + 2 chirp multiplies")

    if cls == "segmented_aperture":
        # NO FLOP FLOOR, deliberately, and flops=0 is what suppresses the ideal
        # line on the figure and the ideal row in the table.
        #
        # Rasterisation has no derivable arithmetic requirement, so anything put
        # here would be invented. An earlier version used N^2 -- one write per
        # output pixel -- and it was worse than nothing on three counts. It is a
        # memory-traffic bound wearing the label that means "the arithmetic the
        # physics requires" everywhere else in this suite. It carries no
        # information of its own, because plots.py anchors the ideal line through
        # the fastest measured point, so only its slope is content and that slope
        # is just N^2. And it does not bound the data: HCIPy runs 36.8 ms at
        # N=256 against 43.5 at N=512, far flatter than N^2, so the line crossed
        # the curves and invited the reading that a code was beating a physics
        # floor.
        #
        # The question that line was meant to answer -- is this code O(N^2) or
        # steeper -- is real, and POPPY is genuinely steeper. Log-log gridlines
        # answer it without asserting a floor that does not exist.
        #
        # `tops` still carries the honest per-pixel count, which is a
        # transcendental/op tally rather than a claim about required arithmetic.
        seg = case.segmented
        n = case.n_pupil
        nseg = seg.n_segments if seg else 0
        return Work(
            0.0, float(n * n), isz // 2 * n * n,
            f"{nseg} segments + {seg.spider_count if seg else 0} spiders onto "
            f"{n}x{n}; no arithmetic floor -- rasterisation is memory-bound and "
            f"any FLOP figure here would be invented, so none is reported"
        )

    if cls == "czt":
        # Bluestein: per 1-D transform, 3 FFTs of length L plus 2 elementwise
        # passes; separable row-column over (N_p + N_f) lines. APPROXIMATE --
        # included because CZT is the theoretical floor for arbitrary output
        # sampling and none of the six codes benchmarked here exposes it, which
        # is itself a result worth quantifying.
        L = next_pow2(n_p + n_f - 1)
        per_line = 3 * fft_1d(L) + 2 * elementwise_mul(L)
        flops = (n_p + n_f) * per_line
        return Work(flops, 0.0, isz * 4 * L * (n_p + n_f),
                    f"Bluestein CZT, L={L} (approximate)")

    raise ValueError(f"no ideal model for algorithm_class {case.algorithm_class!r}")


def basis_work(case: Case, n_modes: int) -> Work:
    """Cost of rendering an n_modes basis onto the pupil, plus the phasor.

    At large parameter counts this dominates everything else: P=1024 modes over
    a 1024^2 grid is ~2 GFLOP, larger than the propagation itself. Whether it
    is hoisted into build() is pinned by case.basis_caching, or the gradient
    board silently becomes a basis-caching benchmark.
    """
    n_p = case.n_pupil
    accum = 2.0 * n_modes * n_p * n_p          # FMA per mode per pixel
    phasor_tops = float(n_p * n_p)             # exp(2j*pi*opd)
    return Work(accum, phasor_tops, itemsize(case.dtype) * n_p * n_p,
                f"{n_modes} modes over {n_p}^2 + phasor")


def gradient_ideal_work(case: Case) -> Work:
    """Reverse-mode floor: backward ~= forward for a linear propagation.

    The adjoint of an MFT is another MFT with conjugate-transposed kernels, so
    the whole gradient should cost about 2x the forward pass REGARDLESS of the
    parameter count P. That P-independence is the entire value proposition of
    reverse mode, and it is what the board's P-sweep is testing.
    """
    p = case.parameters.count if case.parameters else 0
    fwd = ideal_work(case) + basis_work(case, p)
    bwd = ideal_work(case) + basis_work(case, p)
    total = fwd + bwd
    total.detail = f"forward + adjoint (P={p}, expected P-independent)"
    return total


# ------------------------------------------------- efficiency decomposition --
@dataclass
class Efficiency:
    flops_ideal: float
    flops_actual: float | None
    seconds: float
    roofline_flops_per_s: float | None
    algorithmic_overhead: float | None   # A = actual / ideal   (>= 1 expected)
    execution_efficiency: float | None   # E = actual / (t * R) (<= 1 expected)
    overall: float | None                # ideal / (t * R) = E / A

    def to_dict(self) -> dict:
        return asdict(self)


def efficiency(flops_ideal: float, seconds: float,
               flops_actual: float | None = None,
               roofline: float | None = None) -> Efficiency:
    """Split efficiency into the two interpretable factors.

    A answers "is this code doing more arithmetic than the physics requires" --
    hardware-independent, and the closest thing here to a measure of how well
    the fundamental physics is implemented.

    E answers "is it doing that arithmetic well on this machine" -- the backend,
    threading and memory question.

    Ranking on wall time alone conflates the two. A code can lose on time while
    winning on A (right algorithm, poor execution: fixable) or the reverse
    (brute force on a fast backend: architectural).
    """
    A = (flops_actual / flops_ideal) if (flops_actual and flops_ideal) else None
    E = (flops_actual / (seconds * roofline)) if (flops_actual and roofline and seconds) else None
    overall = (flops_ideal / (seconds * roofline)) if (roofline and seconds) else None
    return Efficiency(flops_ideal, flops_actual, seconds, roofline, A, E, overall)
