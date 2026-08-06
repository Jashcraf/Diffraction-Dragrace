"""VizTracer integration.

VizTracer is used for *attribution* -- where the time goes inside a propagation
-- and never for headline numbers. Its overhead is per-Python-function-call, so
a NumPy-vectorised code pays almost nothing while a loop-heavy one pays a lot;
ranking codes on traced wall time would systematically flatter exactly the ones
this suite is trying to compare fairly. Timing and tracing are separate modes on
the same case for that reason, and traced results are stamped `traced: true` and
excluded from every comparison plot.

For JAX-backed adapters VizTracer sees essentially nothing useful -- the whole
propagation is one XLA executable -- so those adapters declare `jax_profiler`
instead. Both emit Chrome/Perfetto-format JSON, so trace_summary.py ingests
either.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


def available() -> bool:
    try:
        import viztracer  # noqa: F401
        return True
    except ImportError:
        return False


@contextmanager
def trace(output: str | Path, max_stack_depth: int = 15, min_duration: float = 0.0,
          ignore_frozen: bool = True):
    """Trace a block to a Chrome-trace JSON file.

    max_stack_depth is capped by default because a full-depth trace of a deep
    call chain produces files large enough to be unusable in the viewer, and
    the interesting attribution lives in the top few frames.
    """
    try:
        from viztracer import VizTracer
    except ImportError:
        yield None
        return

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracer = VizTracer(
        output_file=str(output),
        max_stack_depth=max_stack_depth,
        min_duration=min_duration,
        ignore_frozen=ignore_frozen,
        log_gc=False,
    )
    tracer.start()
    try:
        yield tracer
    finally:
        tracer.stop()
        tracer.save()


@contextmanager
def jax_trace(output: str | Path):
    """jax.profiler equivalent, for adapters whose work happens inside XLA."""
    try:
        import jax
    except ImportError:
        yield None
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(output)):
        yield output
