"""Timing and memory measurement.

Three rules encoded here, each of which exists because breaking it produces a
plausible-looking wrong number:

  1. sync() is inside the clock. Without it an async backend returns before any
     arithmetic has happened and reports a ~100x speedup that is entirely
     dispatch latency.
  2. Timing runs are never traced. VizTracer's overhead is per-Python-call, so
     it penalises loop-heavy codes far more than vectorised ones -- traced wall
     time would systematically flatter exactly the codes this suite is meant to
     compare fairly.
  3. Device->host transfer is measured, but separately. Conflating it with
     compute is how GPU benchmarks lie in both directions.
"""
from __future__ import annotations

import resource
import statistics
import tracemalloc
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class Timing:
    warmup: int
    repeats: int
    device_compute: list[float] = field(default_factory=list)
    host_available: list[float] = field(default_factory=list)
    traced: bool = False
    unit: str = "s"

    def _stats(self, xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        s = sorted(xs)
        n = len(s)
        return {
            "min": s[0],
            "median": statistics.median(s),
            "mean": statistics.fmean(s),
            "p95": s[min(n - 1, int(0.95 * n))],
            "iqr": s[int(0.75 * n) - 1] - s[int(0.25 * n)] if n >= 4 else 0.0,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["device_compute_stats"] = self._stats(self.device_compute)
        d["host_available_stats"] = self._stats(self.host_available)
        return d

    @property
    def warm_median(self) -> float:
        return statistics.median(self.device_compute) if self.device_compute else float("nan")


def time_propagation(adapter, state: Any, warmup: int, repeats: int,
                     traced: bool = False) -> Timing:
    """Warm up, then time `repeats` iterations.

    Warm-up is not politeness: it fills FFTW wisdom, the cuFFT plan cache and
    the NVRTC kernel cache, triggers any remaining JIT, and first-touches the
    output pages. Its cost is real but it is a setup cost, and setup is
    measured separately by setup_cost.py.
    """
    t = Timing(warmup=warmup, repeats=repeats, traced=traced)

    for _ in range(warmup):
        adapter.sync(adapter.propagate(state))

    for _ in range(repeats):
        t0 = perf_counter()
        out = adapter.propagate(state)
        adapter.sync(out)
        t1 = perf_counter()

        t2 = perf_counter()
        adapter.to_host(out)
        t3 = perf_counter()

        t.device_compute.append(t1 - t0)
        t.host_available.append((t1 - t0) + (t3 - t2))
    return t


def time_gradient(adapter, state: Any, warmup: int, repeats: int) -> Timing:
    t = Timing(warmup=warmup, repeats=repeats)
    for _ in range(warmup):
        adapter.sync(adapter.gradient(state))
    for _ in range(repeats):
        t0 = perf_counter()
        out = adapter.gradient(state)
        adapter.sync(out)
        t1 = perf_counter()
        t.device_compute.append(t1 - t0)
        t.host_available.append(t1 - t0)
    return t


# ------------------------------------------------------------------ memory --
@dataclass
class Memory:
    tracemalloc_peak_bytes: int | None = None
    rss_peak_bytes: int | None = None
    device_bytes: int | None = None
    tool: str = "tracemalloc+rusage"

    def to_dict(self) -> dict:
        return asdict(self)


def measure_memory(adapter, state: Any) -> Memory:
    """Single untimed iteration under tracemalloc.

    tracemalloc does see NumPy array data -- NumPy registers its allocations
    under np.lib.tracemalloc_domain -- but it cannot see FFTW or MKL scratch
    buffers, nor anything on a GPU. RSS high-water covers the former; adapters
    that own device memory supply the latter via an optional device_memory()
    method, because JAX and CuPy each need their own accounting and RSS is
    meaningless for both.
    """
    tracemalloc.start()
    try:
        out = adapter.propagate(state)
        adapter.sync(out)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KiB on Linux
    dev_fn = getattr(adapter, "device_memory", None)
    device = dev_fn() if callable(dev_fn) else None

    return Memory(
        tracemalloc_peak_bytes=int(peak),
        rss_peak_bytes=int(rss_kb) * 1024,
        device_bytes=device,
        tool="tracemalloc+rusage" + ("+device" if device is not None else ""),
    )


# -------------------------------------------------------------- sync guard --
def check_sync_scaling(t_small: float, t_large: float,
                       flops_small: float, flops_large: float,
                       min_ratio: float = 1.3) -> tuple[bool, str]:
    """Detect an adapter whose sync() is effectively a no-op.

    A broken sync produces timings that barely move with problem size, because
    what is being measured is dispatch latency rather than arithmetic. Compare
    the measured ratio against the FLOP ratio: anything implausibly flat is
    flagged rather than silently published as a record-breaking result.

    Returns (suspect, message).
    """
    if t_small <= 0 or flops_small <= 0:
        return True, "degenerate timing or FLOP count"
    measured = t_large / t_small
    expected = flops_large / flops_small
    if measured < min_ratio and expected > 2.0:
        return True, (
            f"SYNC SUSPECT: work grew {expected:.1f}x but time grew only "
            f"{measured:.2f}x. The adapter's sync() is probably not blocking, so "
            f"these timings measure dispatch latency, not computation."
        )
    return False, f"sync ok: time {measured:.2f}x for {expected:.1f}x work"


def verify_dtype(arr: np.ndarray, case) -> None:
    """Fail loudly on a precision mismatch.

    JAX defaults to complex64. Benchmarking dLux at single precision against
    POPPY at double is not a comparison, and the failure is silent unless
    something checks.
    """
    got = np.asarray(arr).dtype
    want = np.dtype(case.dtype)
    if got != want:
        raise ValueError(
            f"dtype mismatch: case {case.id!r} specifies {want}, adapter produced {got}. "
            f"For JAX-backed adapters set JAX_ENABLE_X64=1 before the first jax import."
        )
