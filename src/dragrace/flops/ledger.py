"""Kernel-shape ledger: what a library ACTUALLY computed.

Records an ordered list of (op, shape, dtype) for every FFT and GEMM-class call
made during one propagation, and prices each from its observed shape using the
same conventions as model.py.

The FLOP total is useful, but the ordered ledger is the more valuable artifact.
Diffing two adapters on the same case answers "why is A slower than B" directly:

  poppy  @ p2f_fft_n1024_q4:  fft2(4096,4096) x1   ifft2(4096,4096) x1   exp(1024,1024) x1
  prysm  @ p2f_fft_n1024_q4:  fft2(4096,4096) x1                         exp(1024,1024) x1

A stray transform, a rebuilt kernel matrix, an array one power of two larger
than necessary -- these become a structural diff rather than an unexplained
1.8x in a bar chart, which is what makes them filable as issues.

LIMITATION, stated plainly: `A @ B` between plain ndarrays calls
ndarray.__matmul__ in C and does NOT route through np.matmul, so it is
invisible here. Explicit np.matmul / np.dot / np.einsum / np.tensordot calls are
caught. For MFT-heavy codes the ledger therefore under-counts GEMMs, and the
analytic model in model.py remains the primary FLOP source; validate against
hardware counters (`perf stat -e fp_arith_inst_retired.*`, or LIKWID's FLOPS_DP)
once per machine. dLux does not need any of this -- XLA reports exact costs via
compiled.cost_analysis().

Never active during a timing run: the wrappers add per-call overhead, and
`--mode ledger` is a separate pass.
"""
from __future__ import annotations

import functools
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import model


@dataclass
class Entry:
    op: str
    in_shape: tuple
    out_shape: tuple
    dtype: str
    flops: float
    tops: float = 0.0

    def key(self) -> str:
        return f"{self.op}{self.in_shape}"


@dataclass
class Ledger:
    entries: list[Entry] = field(default_factory=list)

    def reset(self) -> None:
        """Discard everything recorded so far.

        Used to drop the calls made during configure()/build() so that only the
        propagation itself is priced -- while still having those run inside the
        patched context, which is what makes an adapter that caches a callable
        (`self._fft2 = np.fft.fft2`) pick up the instrumented version.
        """
        self.entries.clear()

    @property
    def flops(self) -> float:
        return sum(e.flops for e in self.entries)

    @property
    def tops(self) -> float:
        return sum(e.tops for e in self.entries)

    def histogram(self) -> dict[str, int]:
        h: dict[str, int] = {}
        for e in self.entries:
            h[e.key()] = h.get(e.key(), 0) + 1
        return h

    def by_op(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for e in self.entries:
            d[e.op] = d.get(e.op, 0.0) + e.flops
        return d

    def to_dict(self) -> dict:
        return {
            "flops_total": self.flops,
            "tops_total": self.tops,
            "n_calls": len(self.entries),
            "histogram": self.histogram(),
            "flops_by_op": self.by_op(),
            "sequence": [
                {"op": e.op, "in": list(e.in_shape), "out": list(e.out_shape),
                 "dtype": e.dtype, "flops": e.flops}
                for e in self.entries
            ],
            "limitations": [
                "`@` between plain ndarrays bypasses np.matmul and is not counted",
            ] + ([] if self.entries else [
                "no calls were intercepted: either the library issues none (dLux "
                "runs one XLA executable -- read compiled.cost_analysis() instead) "
                "or it bound NumPy's entry points into closures before the "
                "instrumentation was installed. Do not read flops_total=0 as work "
                "not done; re-run with `--mode ledger`, which defers the import "
                "into the patched region."
            ]),
        }

    def render(self) -> str:
        if not self.entries:
            # An empty ledger has two very different causes and the reader
            # cannot tell them apart from a blank table -- which is how HCIPy
            # silently reported 0 GFLOP for a propagation that ran two FFTs.
            # A silent zero is worse than a crash, so say both out loud.
            return (
                "no instrumented calls were intercepted.\n\n"
                "  Either the library issues none -- dLux runs the whole "
                "propagation as a single\n  XLA executable, so this is expected "
                "and `compiled.cost_analysis()` is the\n  number to read -- or it "
                "captured NumPy's entry points into closures before\n  the "
                "instrumentation was installed, in which case the zero is an "
                "artifact.\n  A library must be imported INSIDE record() to be "
                "intercepted; see cmd_ledger."
            )
        lines = [f"{'op':<28} {'shape':<22} {'n':>4} {'GFLOP':>10}"]
        lines.append("-" * 68)
        agg: dict[str, list] = {}
        for e in self.entries:
            a = agg.setdefault(e.key(), [e.op, e.in_shape, 0, 0.0])
            a[2] += 1
            a[3] += e.flops
        for op, shape, n, fl in agg.values():
            lines.append(f"{op:<28} {str(shape):<22} {n:>4} {fl / 1e9:>10.4f}")
        lines.append("-" * 68)
        lines.append(f"{'TOTAL':<28} {'':<22} {len(self.entries):>4} {self.flops / 1e9:>10.4f}")
        return "\n".join(lines)


def _shape(x: Any) -> tuple:
    return tuple(getattr(x, "shape", ()) or ())


def _dtype(x: Any) -> str:
    return str(getattr(x, "dtype", "?"))


def _fft_flops(shape: tuple, axes: int) -> float:
    if not shape:
        return 0.0
    if axes == 1:
        n = shape[-1]
        lines = math.prod(shape[:-1]) if len(shape) > 1 else 1
        return lines * model.fft_1d(n)
    if len(shape) >= 2:
        return model.fft_2d(shape[-2], shape[-1]) * (math.prod(shape[:-2]) or 1)
    return model.fft_1d(shape[-1])


def _preimport_numpy_wrappers() -> None:
    """Import libraries that capture NumPy's entry points AT IMPORT TIME.

    The patches below replace `np.exp` and `np.fft.fft2` with plain Python
    functions. Any third party that inspects those objects while being imported
    must therefore be imported *first*, against pristine NumPy -- otherwise it
    sees the wrapper and fails, somewhere with no connection to the propagator
    under test.

    dask does both of the things that break:

        dask.array.ufunc.ufunc.__init__  asserts isinstance(func, np.ufunc),
                                         which no wrapper can satisfy
        dask.array.fft.fft_wrap          derives its kind from func.__name__
                                         (functools.wraps now keeps that right,
                                         but the ufunc check is unfixable)

    and it is imported transitively by `pyfftw.interfaces`, which both hcipy and
    poppy pull in. That made `dragrace ledger --adapter hcipy` die with
    "TypeError: must be an instance of `ufunc`" -- a message about dask, raised
    from inside HCIPy's import, describing nothing about HCIPy.

    The callers that already import the adapter's library before opening this
    context (worker.run does, via check_requirements) were never affected. This
    makes the context safe on its own rather than relying on call order.
    """
    try:
        import dask.array                              # noqa: F401
    except Exception:                                  # noqa: BLE001
        pass                                           # not installed: nothing to protect


#: The ledger owning the currently-installed patches, if any. Exists so that
#: record() can nest: worker.run() opens the context before it configures the
#: adapter (so the library is imported against instrumented NumPy), and
#: _measure() then opens it again around the propagation without knowing whether
#: it is the outer one.
_ACTIVE: Ledger | None = None


def active() -> Ledger | None:
    return _ACTIVE


@contextmanager
def record(patch_scipy: bool = True):
    """Instrument NumPy/SciPy FFT and GEMM entry points for one computation.

    Reentrant. A nested call yields the ledger the outer one already owns rather
    than patching a second time -- double-wrapping would count every call twice
    and, on exit, restore a wrapper instead of the original function.
    """
    global _ACTIVE
    import numpy as np

    if _ACTIVE is not None:
        yield _ACTIVE
        return

    _preimport_numpy_wrappers()

    led = Ledger()
    patches: list[tuple[Any, str, Any]] = []

    def wrap_fft(mod, fname: str, axes: int):
        orig = getattr(mod, fname, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def wrapper(a, *args, **kwargs):
            out = orig(a, *args, **kwargs)
            led.entries.append(Entry(
                op=f"{mod.__name__}.{fname}", in_shape=_shape(a), out_shape=_shape(out),
                dtype=_dtype(out), flops=_fft_flops(_shape(out), axes)))
            return out

        setattr(mod, fname, wrapper)
        patches.append((mod, fname, orig))

    def wrap_gemm(mod, fname: str):
        orig = getattr(mod, fname, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def wrapper(a, b, *args, **kwargs):
            out = orig(a, b, *args, **kwargs)
            sa, sb, so = _shape(a), _shape(b), _shape(out)
            if len(sa) == 2 and len(sb) == 2:
                fl = model.zgemm(sa[0], sa[1], sb[1])
            else:
                fl = model.ZGEMM_PER_MAC * (math.prod(so) or 1) * (sa[-1] if sa else 1)
            led.entries.append(Entry(f"{mod.__name__}.{fname}", sa, so, _dtype(out), fl))
            return out

        setattr(mod, fname, wrapper)
        patches.append((mod, fname, orig))

    def wrap_transcendental(mod, fname: str):
        orig = getattr(mod, fname, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def wrapper(a, *args, **kwargs):
            out = orig(a, *args, **kwargs)
            n = float(math.prod(_shape(out)) or 1)
            led.entries.append(Entry(f"{mod.__name__}.{fname}", _shape(a), _shape(out),
                                     _dtype(out), 0.0, tops=n))
            return out

        setattr(mod, fname, wrapper)
        patches.append((mod, fname, orig))

    try:
        for f in ("fft", "ifft"):
            wrap_fft(np.fft, f, 1)
        for f in ("fft2", "ifft2", "fftn", "ifftn"):
            wrap_fft(np.fft, f, 2)
        for f in ("matmul", "dot", "tensordot", "einsum"):
            wrap_gemm(np, f)
        for f in ("exp", "cos", "sin"):
            wrap_transcendental(np, f)

        if patch_scipy:
            try:
                import scipy.fft as sfft
                for f in ("fft", "ifft"):
                    wrap_fft(sfft, f, 1)
                for f in ("fft2", "ifft2", "fftn", "ifftn"):
                    wrap_fft(sfft, f, 2)
            except Exception:                        # noqa: BLE001
                pass
        _ACTIVE = led
        yield led
    finally:
        _ACTIVE = None
        for mod, fname, orig in reversed(patches):
            setattr(mod, fname, orig)
