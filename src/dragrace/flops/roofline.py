"""Roofline: measured machine peaks, and which bound a computation sits under.

Peaks are measured rather than taken from a spec sheet. Theoretical peak FLOP/s
is not a bound any of these codes could reach, so scoring against it would make
every adapter look uniformly terrible and hide the differences between them. A
large zgemm is what "as fast as this machine gets at complex arithmetic"
actually means in practice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .model import Work


@dataclass
class Machine:
    peak_flops_per_s: float
    peak_bandwidth_bytes_per_s: float
    gemm_size: int
    threads_seen: dict
    note: str = ""

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity above which a computation is compute-bound."""
        return self.peak_flops_per_s / self.peak_bandwidth_bytes_per_s

    def bound(self, work: Work) -> float:
        """Applicable roofline: min(peak, AI * bandwidth)."""
        return min(self.peak_flops_per_s,
                   work.arithmetic_intensity * self.peak_bandwidth_bytes_per_s)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ridge_point_flops_per_byte"] = self.ridge_point
        return d


def measure_peak_flops(n: int = 2048, repeats: int = 3) -> float:
    """Sustained complex128 GEMM rate: 8*n^3 flops per call."""
    a = (np.random.rand(n, n) + 1j * np.random.rand(n, n)).astype(np.complex128)
    b = (np.random.rand(n, n) + 1j * np.random.rand(n, n)).astype(np.complex128)
    a @ b                                            # warm up / first-touch
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        a @ b
        best = min(best, perf_counter() - t0)
    return (8.0 * n ** 3) / best


def measure_bandwidth(n: int = 32_000_000, repeats: int = 3) -> float:
    """STREAM triad: a = b + s*c, 3 arrays touched per element (2 read 1 write)."""
    b = np.ones(n, dtype=np.float64)
    c = np.ones(n, dtype=np.float64)
    a = np.empty(n, dtype=np.float64)
    np.add(b, 3.0 * c, out=a)
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        np.add(b, 3.0 * c, out=a)
        best = min(best, perf_counter() - t0)
    return (3.0 * n * 8.0) / best


def measure_machine(gemm_size: int = 2048, quick: bool = False) -> Machine:
    from ..backend import detect_thread_counts

    n = 1024 if quick else gemm_size
    return Machine(
        peak_flops_per_s=measure_peak_flops(n),
        peak_bandwidth_bytes_per_s=measure_bandwidth(4_000_000 if quick else 32_000_000),
        gemm_size=n,
        threads_seen=detect_thread_counts(),
        note="measured zgemm + STREAM triad; not theoretical peak",
    )


def classify(work: Work, machine: Machine) -> dict:
    ai = work.arithmetic_intensity
    bound = machine.bound(work)
    return {
        "arithmetic_intensity": ai,
        "ridge_point": machine.ridge_point,
        "bound": "compute" if ai >= machine.ridge_point else "bandwidth",
        "bound_flops_per_s": bound,
    }
