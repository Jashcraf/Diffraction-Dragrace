#!/usr/bin/env python
"""Where one pupil-to-focus matrix DFT actually spends its time, per code.

The sibling of plot_free_space_breakdown.py, asked of the other algorithm class.
That figure took one free-space propagation apart and found that the transform
every code has to do is the same size for everybody, and that what separates the
bars is what each library does *around* it. This one asks whether that holds
where the required work is two BLAS matrix products instead of two FFTs.

One stacked bar per code, split into

    GEMM        the two matrix products -- the only work the physics requires
    DFT matrix  rebuilding the (N_f, N_p) basis matrices, per call
    pupil pass  whole-pupil-sized work: the wavefront rebuilt per call, casts,
                scratch, normalisation sweeps
    other       everything else: object construction, unit checks, dispatch

Every code here computes the same quantity to the same tolerance -- the case
gates at 1e-10 and each adapter is re-gated in this script rather than trusted
to have passed elsewhere (`rel_l2` in the per-bar JSON, worst value in the
caption). So the bars are cost, not a quality trade.

HOW THE SPLIT IS MEASURED. The blue segment is TIMED, not attributed: every
matrix-product entry point these five NumPy-backed codes use is wrapped with a
stopwatch before any of them is imported, and the profile is left to apportion
only what is left over. That is the correction plot_free_space_breakdown.py
already carries, applied here from the start -- deriving the segment from a
profile share instead once put a wrong ordering on the published figure.

Reaching every code's GEMM takes three interceptions, because they do not agree
on how to ask for one:

    np.dot                 POPPY (matrixDFT.py:374), lentil (fourier.py:97)
    scipy.linalg.blas.zgemm  HCIPy, which calls BLAS by hand on transposed
                           F-ordered views to avoid a copy
    the `@` operator       numpy_baseline, prysm (fttools.py MDFT.__call__),
                           and lentil's inner E1.dot(f)

`@` is the awkward one: it goes through ndarray.__matmul__ in C and never looks
at np.matmul, so a module-level patch cannot see it. _TimedMatrix solves it from
the other side -- one OPERAND is viewed as an ndarray subclass, which makes
Python dispatch the operator into Python code where a clock can be started. The
view shares the buffer and the strides and is stripped again before the BLAS
call, so the arithmetic is identical; the cost is a few microseconds of Python
per propagation, and the child measures the median with the instrumentation
removed as well and reports both (`instrument_overhead` in the JSON).

WHAT MAKES THE BLUE SEGMENTS COMPARABLE. Each intercepted call also records its
operand shapes, so the FLOPs each code spends in BLAS are counted rather than
assumed: 8*m*k*n summed over the calls, against the case's ideal
8*N_p*N_f*(N_p+N_f). A code that reaches the same answer by doing more
arithmetic shows up as a larger count, not as a mysteriously taller bar.

The rest of each bar is cProfile self-time bucketed by an explicit rule table
(RULES below, first match wins), rescaled onto the measured median. Profiling
inflates the Python-heavy adapters most, which is exactly the axis this figure
compares, so the profiler says where the time went and never how much of it
there was.

A JAX adapter cannot be profiled that way -- it is one fused XLA program and
cProfile sees only block_until_ready -- so its bar is bracketed and hatched, as
dLux's is on the free-space figure:

    GEMM        the same two matrix products in XLA with both matrices held
                constant, i.e. the XLA floor for this case's arithmetic. Not
                OpenBLAS: the JAX bars' blue is a different implementation and
                is not comparable with the NumPy codes' blue.
    DFT matrix  the increment when dLux's transfer matrices are instead built
                from the traced wavelength, which is what dLux's own program
                does. Measured against a rung that reproduces dLux's output to
                ~1e-15 (`rel_hoisted_vs_full`), so it is a decomposition and
                not a fit. Zero for a JAX adapter that hoists its kernels into
                build(), by construction.
    pupil pass  the remainder. A compiled call has no per-call Python, so the
                JAX bars carry no `other`.

Pinned to one core with sched_setaffinity, the same mechanism and for the same
reason as worker._pin_cpus: XLA honours no thread environment variable. Each
adapter is measured in its own subprocess, because they mutate global state to
configure themselves and in one process the last one configured would set the
terms for the rest.

    python scripts/plot_mft_breakdown.py [--reps 9] [--rounds 5]

Run it with the interpreter of the `dragrace` environment; children inherit it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "cases" / "pupil_to_focus" / "mft_n1024_q4.yaml"
CONFIG = "cpu_numpy_1t"
OUT = ROOT / "docs" / "figures" / "mft_pupil_to_focus_breakdown.png"
#: Every number the figure draws, written beside it. Two reasons: a layout fix
#: should not cost another ten minutes of measurement (--redraw), and a reader
#: who wants to check a segment against the shapes and call counts it came from
#: should not have to rerun the machine to see them.
DATA = OUT.with_suffix(".json")

#: Left to right: the three codes that hoist their kernels, the two that rebuild
#: them per call, then the two JAX programs. Fixed rather than sorted by runtime,
#: so the figure keeps its shape when a code is added or drops off the board.
ADAPTERS = ("numpy_baseline", "hcipy", "prysm", "lentil", "poppy", "dlux")

BUCKETS = ("gemm", "kernel", "pupil", "other")
LABELS = {
    "gemm": "GEMM — the two matrix products, the only required work",
    "kernel": "DFT basis matrices, rebuilt every call",
    "pupil": "whole-pupil passes: wavefront rebuilt per call, casts, scratch",
    "other": "other per-call overhead",
}
#: Same palette and the same meaning as the free-space breakdown, so the two
#: figures can be read side by side. The residual is inert grey so it never
#: reads as a finding; every segment is value-labelled, so nothing rests on hue.
COLORS = {"gemm": "#2a78d6", "kernel": "#eda100",
          "pupil": "#e34948", "other": "#b3b1ac"}

INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

# --------------------------------------------------------------- bucketing --
#: (substring of "file:lineno(function)", bucket). First match wins, so the
#: specific rules come before the general ones. Anything unmatched is `other`,
#: and --verbose prints every function above the noise floor with the bucket it
#: was given, so a wrong rule shows up as a wrong label rather than as a
#: plausible bar.
RULES = [
    # the matrix products. These frames are this script's own stopwatches plus
    # whatever slipped past them; their tottime is dropped from the
    # apportionment entirely, because the blue segment is measured separately.
    ("(_gemm_np)", "gemm"),
    ("(_gemm_blas)", "gemm"),
    ("(_gemm_op)", "gemm"),
    ("(__matmul__)", "gemm"),
    ("(__rmatmul__)", "gemm"),
    ("(dot)", "gemm"),
    ("multiarray.dot", "gemm"),
    ("method 'dot' of 'numpy", "gemm"),
    ("fblas", "gemm"),
    # per-call construction of the (N_f, N_p) basis matrices and the coordinate
    # vectors that exist only to feed them
    ("(_dft2_matrices)", "kernel"),                  # lentil
    ("(_dft2_coords)", "kernel"),
    ("necompiler.py", "kernel"),                     # POPPY + HCIPy: numexpr exp
    ("matrixDFT.py", "kernel"),                      # POPPY: outer products, exp
    ("(_compute_matrices)", "kernel"),               # HCIPy
    ("(outer)", "kernel"),
    ("(fftrange)", "kernel"),
    ("(coordinates_for_focus)", "kernel"),
    # whole-pupil-sized passes
    ("(__mul__)", "pupil"),                          # lentil: Wavefront * Pupil
    ("(_mul_array)", "pupil"),                       # lentil: the amplitude/OPD apply
    ("(__imul__)", "pupil"),                         # POPPY: optic applied to wavefront
    ("(normalize)", "pupil"),                        # POPPY: whole-array renormalisation
    ("method 'reduce' of 'numpy.ufunc'", "pupil"),   # the sum inside it
    ("(pad_to_oversample)", "pupil"),
    ("(_resample_wavefront)", "pupil"),
    ("method 'astype'", "pupil"),
    ("method 'copy' of 'numpy", "pupil"),
    ("built-in method numpy.array", "pupil"),
    ("(zeros)", "pupil"),
    ("(ones)", "pupil"),
    ("(empty)", "pupil"),
    ("method 'reshape'", "pupil"),
    ("method 'ravel'", "pupil"),
    ("(asarray)", "pupil"),
    ("(_dot_wavefront)", "pupil"),
    ("(multiply)", "pupil"),
]


# ------------------------------------------------------- GEMM interception --
#: (seconds, shape_a, shape_b) for every matrix product in the region currently
#: being measured. See the module docstring for why this is timed rather than
#: read off a profile.
_GEMM_CALLS: list[tuple] = []
#: Unwrapped originals, so the stopwatch never times itself.
_ORIG: dict = {}
#: Callables that put the process back the way it was found, so the bar height
#: can be re-measured with none of this installed.
_UNDO: list = []


def _shp(x):
    return tuple(getattr(x, "shape", ()) or ())


def _record(dt, a, b):
    _GEMM_CALLS.append((dt, _shp(a), _shp(b)))


def _gemm_op(a, b, out=None):
    """Timed matmul for the `@` operator and ndarray.dot.

    Deliberately a Python function: cProfile can see it, so RULES can bucket it
    to `gemm` and the apportionment can drop it. A C-level `@` would instead
    charge its time to whichever adapter frame called it, and land silently in
    `other`.
    """
    import numpy as np

    mm = _ORIG["matmul"]
    a = np.asarray(a)
    b = np.asarray(b)
    t = time.perf_counter()
    r = mm(a, b) if out is None else mm(a, b, out=out)
    _record(time.perf_counter() - t, a, b)
    return r


def _install_gemm_timers() -> None:
    """Wrap every matrix-product entry point these codes use.

    Must run before the propagators are imported, for the same reason the FFT
    timers in the free-space script must: a library that binds `np.dot` into a
    module-level name at import would hold the unwrapped function and report
    zero. HCIPy is the case in point -- it resolves `blas.zgemm` per call, but
    `from scipy.linalg import blas` happens at import.
    """
    import numpy as np

    _ORIG["matmul"] = np.matmul
    _ORIG["dot"] = np.dot

    def wrap(mod, attr):
        fn = getattr(mod, attr)

        def _gemm_np(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            dt = time.perf_counter() - t
            _record(dt, a[0] if a else None, a[1] if len(a) > 1 else None)
            return out

        setattr(mod, attr, _gemm_np)
        _UNDO.append(lambda mod=mod, attr=attr, fn=fn: setattr(mod, attr, fn))

    for a in ("dot", "matmul", "inner", "tensordot", "einsum"):
        if hasattr(np, a):
            wrap(np, a)

    try:
        from scipy.linalg import blas
    except ImportError:
        return

    def wrap_blas(attr):
        fn = getattr(blas, attr)

        def _gemm_blas(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            dt = time.perf_counter() - t
            # gemm(alpha, a, b, ...): the operands are the second and third
            # positional arguments, not the first.
            _record(dt, a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else None)
            return out

        setattr(blas, attr, _gemm_blas)
        _UNDO.append(lambda attr=attr, fn=fn: setattr(blas, attr, fn))

    for a in ("zgemm", "cgemm", "dgemm", "sgemm"):
        if hasattr(blas, a):
            wrap_blas(a)


def _timed_matrix_class():
    """ndarray view that times any matrix product it takes part in.

    `A @ B` on plain ndarrays is dispatched by ndarray.__matmul__ in C and never
    consults np.matmul, so no module-level patch can see it. Viewing one operand
    -- the precomputed DFT basis matrix, which is present in every product these
    codes form -- as a subclass moves the dispatch into Python: Python tries a
    subclass operand's reflected method first, so the matrix catches the product
    whether it sits on the left or the right.

    The view shares buffer, dtype and strides with the original, and _gemm_op
    calls np.asarray before the multiply, so BLAS receives exactly what it would
    have received.
    """
    import numpy as np

    class _TimedMatrix(np.ndarray):
        def __matmul__(self, other):
            return _gemm_op(self, other)

        def __rmatmul__(self, other):
            return _gemm_op(other, self)

        def dot(self, b, out=None):
            return _gemm_op(self, b, out=out)

    return _TimedMatrix


def _instrument(name: str, state: dict) -> str:
    """Point the matmul stopwatch at the operands this code actually uses.

    Returns a one-line description of what was instrumented, printed under
    --verbose so a code whose internals moved shows up as an unmeasured GEMM
    rather than as a quietly missing segment.
    """
    import numpy as np

    tm = _timed_matrix_class()

    if name == "numpy_baseline":
        kx = state["kx"]
        state["kx"] = kx.view(tm)
        _UNDO.append(lambda: state.__setitem__("kx", kx))
        return "numpy_baseline: state['kx'] viewed (used on both sides of `@`)"

    if name == "prysm":
        ex = state["executor"]
        old_x, old_y = ex.Ex, ex.Ey
        ex.Ex, ex.Ey = old_x.view(tm), old_y.view(tm)

        def undo(ex=ex, old_x=old_x, old_y=old_y):
            ex.Ex, ex.Ey = old_x, old_y

        _UNDO.append(undo)
        return "prysm: MDFT.Ex/.Ey viewed (fttools.py MDFT.__call__ uses `@`)"

    if name == "lentil":
        # lentil rebuilds E1/E2 on every call (_dft2_coords is lru_cached, the
        # exponentials are not), so there is nothing on `state` to view. The
        # factory is wrapped instead: E1 meets the field through E1.dot(f), E2
        # through np.dot, which is already wrapped.
        import lentil.fourier as lf

        orig = lf._dft2_matrices

        def patched(*a, **k):
            e1, e2 = orig(*a, **k)
            return e1.view(tm), e2

        lf._dft2_matrices = patched
        _UNDO.append(lambda: setattr(lf, "_dft2_matrices", orig))
        return "lentil: _dft2_matrices patched to hand back a viewed E1"

    return f"{name}: module-level wrappers only (np.dot / scipy blas)"


def _uninstall() -> None:
    while _UNDO:
        _UNDO.pop()()


def bucket_of(key: str) -> str:
    for sub, b in RULES:
        if sub in key:
            return b
    return "other"


def _median(fn, reps: int) -> float:
    import numpy as np

    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def _accuracy(ad, state, result, case) -> float:
    """Re-gate this adapter here rather than trust that it passed elsewhere.

    Mirrors worker.py exactly: complex_field() rather than to_host(), the
    adapter's own declared centring, and compare_intensity for a code whose
    documented entry point returns a PSF.
    """
    import numpy as np
    from dragrace import validate
    from dragrace.reference import reference_field

    centering = getattr(ad, "grid_centering", "pixel")
    quantity = getattr(ad, "output_quantity", "field")
    cmp_fn = validate.compare_intensity if quantity == "intensity" else validate.compare
    out = ad.complex_field(state, result)
    return float(cmp_fn(np.asarray(out), reference_field(case, centering), case).rel_l2)


# ------------------------------------------------------------- measurement --
def measure(name: str, reps: int, verbose: bool) -> dict:
    """Run one adapter and return its bar. Called in the child process."""
    import cProfile
    import pstats

    _install_gemm_timers()          # BEFORE any propagator is imported
    sys.path.insert(0, str(ROOT / "src"))
    from dragrace.case import Case
    from dragrace.config import load_configs
    import dragrace.adapter as adapters

    adapters.discover(str(ROOT / "adapters"))
    case = Case.from_yaml(CASE).scan_cases()[0]
    config = load_configs(str(ROOT / "configs"))[CONFIG]

    ad = adapters.get(name)
    ok = ad.configure(config)
    if ok is not True:
        return {"adapter": name, "skipped": str(ok)}
    sup = ad.supports(case, config)
    if sup is not True:
        return {"adapter": name, "skipped": str(sup)}
    state = ad.build(case, config)

    def one():
        result = ad.propagate(state)
        ad.sync(result)
        return result

    rel = _accuracy(ad, state, one(), case)

    if name in ("dlux", "abcdlux"):
        out = _measure_jax(name, ad, state, case, reps, one)
        out["rel_l2"] = rel
        return out

    import numpy as np

    note = _instrument(name, state)

    # The matrix products, timed per call in the same unprofiled region that
    # produces the bar height -- so the blue segment is a measurement rather
    # than a share of one.
    per_rep = []
    for _ in range(3):
        one()
    for _ in range(reps):
        _GEMM_CALLS.clear()
        one()
        per_rep.append(list(_GEMM_CALLS))
    median = _median(one, reps)
    gemm_s = float(np.median([sum(c[0] for c in r) for r in per_rep]))
    calls = per_rep[-1]
    # 8 real FLOPs per complex multiply-add, summed over the calls actually
    # made: the count that says whether two codes' blue segments are answering
    # the same question.
    gflop = sum(8.0 * c[1][0] * c[1][1] * c[2][1]
                for c in calls if len(c[1]) == 2 and len(c[2]) == 2) / 1e9

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        one()
    pr.disable()

    buckets = dict.fromkeys(BUCKETS, 0.0)
    rows = []
    for func, (_cc, nc, tt, _ct, _callers) in pstats.Stats(pr).stats.items():
        key = f"{func[0]}:{func[1]}({func[2]})"
        b = bucket_of(key)
        buckets[b] += tt
        if tt / reps > 0.0002:
            rows.append((tt / reps, b, nc / reps, key.split("site-packages/")[-1]))

    # What the instrumentation costs, measured rather than asserted: the bar
    # height is the median with all of it removed.
    _uninstall()
    plain = _median(one, reps)

    # The profile now apportions only what is left after the matrix products.
    # Its own gemm tottime is dropped rather than reused: it is the same
    # quantity, measured worse.
    rest = {k: v for k, v in buckets.items() if k != "gemm"}
    rest_total = sum(rest.values()) or 1.0
    # A code that is nothing but its matrix products can time them at fractionally
    # MORE than the whole propagation, because the two medians come from
    # different rep loops and the difference is under a percent. Clamped rather
    # than left to overflow the bar: the conservative direction is to say the
    # propagation is entirely GEMM, never to invent non-GEMM time. The unclamped
    # value is kept in the JSON so the clamp is visible.
    gemm_raw, gemm_s = gemm_s, min(gemm_s, plain)
    room = max(plain - gemm_s, 0.0)
    out = {"adapter": name, "median_s": plain, "gemm_measured": True,
           "gemm_raw_s": gemm_raw, "clamped": gemm_raw > plain,
           "n_gemm_calls": len(calls), "gemm_gflop": gflop, "rel_l2": rel,
           "gemm_shapes": [[c[1], c[2]] for c in calls],
           "instrument_overhead": median / plain if plain else float("nan"),
           "instrumented": note,
           "buckets_s": dict({"gemm": gemm_s},
                             **{k: v / rest_total * room for k, v in rest.items()})}
    if verbose:
        out["rows"] = sorted(rows, reverse=True)[:14]
    return out


def _measure_jax(name: str, ad, state, case, reps: int, one) -> dict:
    """The JAX bracket. See the module docstring for why it is not a profile."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    npx, nf = case.n_pupil, case.n_focus
    rng = np.random.default_rng(0)

    def cplx(shape):
        return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    # The XLA floor for this case's arithmetic: the same two products, both
    # matrices constant, nothing else in the program. The two libraries
    # ASSOCIATE the triple product differently -- dLux forms (tf.T @ phasor) @ tf
    # and abcdLux forms Ky @ (u @ Kx.T) -- which is the same FLOP count through a
    # differently shaped intermediate, and XLA does not cost the two identically.
    # Each bar gets its own library's association; using one for both left
    # abcdLux with a floor 4% above its own measured propagation.
    u, a, b = cplx((npx, npx)), cplx((nf, npx)), cplx((npx, nf))
    prog = (lambda x: (a @ x) @ b) if name == "dlux" else (lambda x: a @ (x @ b))
    base = jax.jit(prog).lower(u).compile()
    jax.block_until_ready(base(u))
    t_gemm = _median(lambda: jax.block_until_ready(base(u)), reps)

    median = _median(one, reps)
    kernel, rel_hoisted = 0.0, None

    if name == "dlux":
        # dLux rebuilds both transfer matrices on every call, because
        # dlu.MFT vmaps get_tf_mat over the traced wavelength. The increment is
        # measured against a rung that is dLux's own call chain with only those
        # two matrices and the normalisation constant hoisted -- reconstructed
        # from optical_systems.py rather than reimplemented, and checked against
        # the real program's output below.
        import dLux.utils as dlu
        from dLux.utils.propagation import transfer_matrix, calc_nfringes

        optics = state["optics"]
        wl = case.wavelength_m
        ps_in = optics.initialise_wavefront(wl, None).pixel_scale
        ps_out = dlu.arcsec2rad(optics.psf_pixel_scale / optics.oversample)
        n_out = optics.psf_npixels * optics.oversample
        # shift = 0 on both axes, so dLux's vmap over `shift` produces two
        # identical matrices; building one and reusing it is exact.
        tf = transfer_matrix(wl, case.n_pupil, ps_in, n_out, ps_out,
                             0.0, None, 0.0, False)
        nfr = calc_nfringes(wl, case.n_pupil, ps_in, n_out, ps_out, None)
        norm = jnp.exp(jnp.log(nfr) - (jnp.log(case.n_pupil) + jnp.log(n_out)))

        def hoisted(w, optics=optics, tf=tf, norm=norm):
            wf = optics.initialise_wavefront(w, None)
            for layer in list(optics.layers.values()):
                wf = layer(wf)
            return jnp.abs(((tf.T @ wf.phasor) @ tf) * norm) ** 2

        arg = state["wavelength"]
        fn_h = jax.jit(hoisted).lower(arg).compile()
        oh = np.asarray(fn_h(arg))
        od = np.asarray(state["fn"](arg))
        rel_hoisted = float(np.linalg.norm(oh - od) / np.linalg.norm(od))
        t_hoisted = _median(lambda: jax.block_until_ready(fn_h(arg)), reps)
        kernel = max(median - t_hoisted, 0.0)

    raw, t_gemm = t_gemm, min(t_gemm, median)
    pupil = max(median - t_gemm - kernel, 0.0)
    return {"adapter": name, "median_s": median, "bracketed": True,
            "rel_hoisted_vs_full": rel_hoisted, "gemm_raw_s": raw,
            "clamped": raw > median,
            "buckets_s": {"gemm": t_gemm, "kernel": kernel,
                          "pupil": pupil, "other": 0.0}}


# ------------------------------------------------------------------ driver --
def _one_round(name: str, reps: int, verbose: bool) -> dict | None:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", JAX_ENABLE_X64="1")
    cmd = [sys.executable, __file__, "--measure", name, "--reps", str(reps)]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = next((ln for ln in reversed(proc.stdout.splitlines())
                 if ln.startswith("{")), None)
    if line is None:
        print(f"  {name}: no result\n{proc.stdout[-1200:]}{proc.stderr[-1200:]}")
        return None
    return json.loads(line)


def collect(reps: int, rounds: int, verbose: bool) -> list[dict]:
    """Measure every adapter `rounds` times and keep its least contaminated run.

    Rounds are interleaved rather than repeated per adapter, so a slow patch of
    machine time cannot land entirely on one bar; the round with the smallest
    median is kept whole, because noise adds time and never removes it (the same
    argument report.best_points makes for a re-measured scan point), and taking
    each bucket's minimum separately would mix rounds and could leave the
    segments not summing to any measured total.
    """
    best: dict[str, dict] = {}
    spread: dict[str, list[float]] = {}
    for r in range(rounds):
        for name in ADAPTERS:
            bar = _one_round(name, reps, verbose and r == 0)
            if bar is None or bar.get("skipped"):
                if bar and bar.get("skipped"):
                    print(f"  {name}: skipped -- {bar['skipped']}")
                continue
            spread.setdefault(name, []).append(bar["buckets_s"]["gemm"])
            cur = best.get(name)
            if cur is None or bar["median_s"] < cur["median_s"]:
                best[name] = bar
        print(f"  -- round {r + 1}/{rounds} done")

    bars = []
    for name in ADAPTERS:
        if name not in best:
            continue
        bar = best[name]
        # Carried onto the figure: the span of this code's OWN matrix-product
        # time across rounds is what says whether the differences between codes
        # mean anything, and it must not be something the reader takes on faith.
        bar["gemm_span_s"] = [min(spread[name]), max(spread[name])]
        bars.append(bar)
        b = bar["buckets_s"]
        lo, hi = bar["gemm_span_s"]
        print(f"  {bar['adapter']:15s} {bar['median_s']*1e3:8.2f} ms   " +
              "  ".join(f"{k} {b[k]*1e3:7.2f}" for k in BUCKETS) +
              f"   gemm over rounds {lo*1e3:6.2f}-{hi*1e3:6.2f}"
              f"   rel_l2 {bar.get('rel_l2', float('nan')):.1e}")
        if bar.get("clamped"):
            print(f"      NOTE: timed GEMM {bar['gemm_raw_s']*1e3:.2f} ms exceeded "
                  f"the propagation median; clamped to it")
        if bar.get("gemm_gflop"):
            print(f"      {bar['n_gemm_calls']} GEMM calls, "
                  f"{bar['gemm_gflop']:.3f} GFLOP, shapes {bar['gemm_shapes']}, "
                  f"instrumentation x{bar['instrument_overhead']:.4f}")
            print(f"      {bar['instrumented']}")
        if bar.get("rel_hoisted_vs_full") is not None:
            print(f"      hoisted rung reproduces the full program to "
                  f"{bar['rel_hoisted_vs_full']:.1e}")
        for row in bar.get("rows", []):
            print(f"      {row[0]*1e3:8.3f} ms {row[2]:5.1f}x [{row[1]:6s}] {row[3][:88]}")
    return bars


def draw(bars: list[dict], case_id: str, npx: int, nf: int, reps: int,
         rounds: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12.2, 8.4), dpi=190)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    tallest = max(b["median_s"] for b in bars) * 1e3
    xs = range(len(bars))
    for x, bar in zip(xs, bars):
        bottom = 0.0
        for key in BUCKETS:
            h = bar["buckets_s"][key] * 1e3
            if h <= 0:
                continue
            ax.bar(x, h, bottom=bottom, width=0.62, color=COLORS[key],
                   edgecolor=SURFACE, linewidth=1.1, zorder=3,
                   hatch="//" if bar.get("bracketed") else None)
            # Value inside the segment while it fits; the whole point of the
            # figure is the size of each piece, and a reader should not have to
            # measure it off the axis.
            if h > 0.055 * tallest:
                ax.text(x, bottom + h / 2, f"{h:.1f}",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", zorder=4,
                        # the residual bucket is a pale grey; white on it is
                        # unreadable
                        color=INK if key == "other" else "white")
            bottom += h
        ax.text(x, bottom + 0.012 * tallest, f"{bar['median_s']*1e3:.1f} ms",
                ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="bold", zorder=4)

    # A BAND, not a floor line, for the same reason the free-space figure draws
    # one: it spans the matrix-product times of the codes that call the same
    # BLAS, and whether their ORDER inside it is resolved is a question the
    # measurement answers rather than the drawing. The verdict sentence below is
    # computed from the round-to-round spans, not asserted.
    nb = [b for b in bars if not b.get("bracketed")]
    lo = min(b["buckets_s"]["gemm"] for b in nb) * 1e3
    hi = max(b["buckets_s"]["gemm"] for b in nb) * 1e3
    swing = max((max(b["gemm_span_s"]) - min(b["gemm_span_s"])) * 1e3
                for b in nb if b.get("gemm_span_s"))
    ax.axhspan(lo, hi, color=INK_SOFT, alpha=0.13, lw=0, zorder=2)
    ax.axhline(lo, color=INK_SOFT, lw=0.8, ls=(0, (5, 3)), zorder=2)
    ax.axhline(hi, color=INK_SOFT, lw=0.8, ls=(0, (5, 3)), zorder=2)
    # Whether the ORDER inside the band means anything is decided by the
    # measurement, not by the drawing: the spread between codes is compared with
    # the largest swing any one code showed across rounds. One round cannot
    # answer it and does not pretend to.
    if rounds < 2:
        verdict = "One round: the order\n  inside it is untested."
    elif (hi - lo) > swing:
        verdict = (f"The {(hi - lo) / lo * 100:.1f}% spread between\n"
                   f"  them is wider than any\n  one code's "
                   f"{swing / lo * 100:.1f}% swing\n  across rounds.")
    else:
        verdict = (f"Their order inside it is\n  not resolved: one code's\n"
                   f"  own swing across rounds\n  is {swing / lo * 100:.1f}%, "
                   f"wider than the\n  {(hi - lo) / lo * 100:.1f}% spread.")
    ax.set_xlim(-0.62, len(bars) - 1 + 0.62 + 1.55)
    ax.text(len(bars) - 1 + 0.52, hi + 0.012 * tallest,
            f"  the two matrix products:\n  {lo:.1f}-{hi:.1f} ms across the\n"
            f"  {len(nb)} codes that call the\n  same OpenBLAS zgemm.\n  {verdict}",
            ha="left", va="bottom", fontsize=8.5, color=INK_SOFT, style="italic",
            linespacing=1.45)

    # Headroom for the legend, which sits at the top left: without it the third
    # entry lands on the total label of whichever bar is fourth from the left.
    ax.set_ylim(0, tallest * 1.32)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b["adapter"] + ("*" if b.get("bracketed") else "")
                        for b in bars], fontsize=10.5, color=INK)
    ax.set_ylabel("runtime per propagation  (ms, one core)", fontsize=10.5, color=INK)
    ax.set_title(f"One pupil-to-focus matrix DFT, {npx}x{npx} -> {nf}x{nf} "
                 f"complex128 — where the time goes",
                 fontsize=13, color=INK, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9.5)

    ax.legend(handles=[Patch(facecolor=COLORS[k], label=LABELS[k]) for k in BUCKETS],
              loc="upper left", frameon=False, fontsize=9.2, labelcolor=INK)

    # Every claim in the caption that is a number is computed here, so a rerun
    # that lands somewhere else rewrites the sentence instead of contradicting
    # it.
    ideal = 8.0 * npx * nf * (npx + nf) / 1e9
    flops = {b["adapter"]: b.get("gemm_gflop") for b in nb}
    same = all(abs(v - ideal) / ideal < 0.02 for v in flops.values() if v)
    worst_rel = max(b.get("rel_l2") or 0.0 for b in bars)
    worst_inst = max(abs(b.get("instrument_overhead", 1.0) - 1.0) for b in nb)
    br = [b["adapter"] for b in bars if b.get("bracketed")]
    jax_note = (
        f"{br[0]}'s bar is bracketed, not profiled: it is one fused XLA program, "
        f"so its GEMM segment is the same two products in XLA with both matrices "
        f"held constant, and its kernel segment is the increment when it rebuilds "
        f"its transfer matrices from the traced wavelength -- measured against a "
        f"rung that reproduces its output to 1e-15, not fitted. XLA's complex128 "
        f"GEMM is not OpenBLAS's, so its blue is not comparable with the other "
        f"{len(nb)}. A compiled call has no per-call Python, which is why it "
        f"carries no grey."
        if len(br) == 1 else
        f"The {len(br)} JAX bars are bracketed, not profiled: each is one fused "
        f"XLA program, so their GEMM segment is the same two products in XLA with "
        f"both matrices constant and their kernel segment is the increment when "
        f"dLux rebuilds its transfer matrices from the traced wavelength. XLA's "
        f"complex128 GEMM is not OpenBLAS's, so their blue is not comparable with "
        f"the other {len(nb)}. A compiled call has no per-call Python, which is "
        f"why they carry no grey.")
    flop_note = (
        f"All {len(nb)} issue the same {nb[0]['n_gemm_calls']} calls for the same "
        f"{ideal:.3f} GFLOP, counted from the operand shapes, so their blue "
        f"segments answer the same question."
        if same else
        "They do NOT all spend the same arithmetic: " +
        ", ".join(f"{k} {v:.3f}" for k, v in flops.items() if v) +
        f" GFLOP against the case's ideal {ideal:.3f}, so the blue segments are "
        f"not directly comparable.")
    caption = "\n".join(
        line for para in (
            f"{case_id} ({npx}^2 pupil, {nf}^2 focal grid at q=4), {CONFIG}, "
            f"complex128, one core (sched_setaffinity, as worker._pin_cpus "
            f"does). Median of {reps} calls, smallest of {rounds} interleaved "
            f"rounds, measured with no profiler and no instrumentation "
            f"attached. Every bar is re-gated against the case's reference here "
            f"rather than trusted to have passed elsewhere: worst rel_l2 "
            f"{worst_rel:.1e} against a 1e-10 gate, so this is cost and not "
            f"accuracy.",
            f"THE BLUE SEGMENT IS TIMED, NOT ATTRIBUTED. Every matrix-product "
            f"entry point is wrapped before any propagator is imported -- np.dot "
            f"for POPPY and lentil, scipy.linalg.blas.zgemm for HCIPy, and the "
            f"`@` operator for numpy_baseline, prysm and lentil's inner product, "
            f"caught by viewing one operand as an ndarray subclass since `@` "
            f"never consults np.matmul. {flop_note} Re-timing each code with all "
            f"of the instrumentation removed moves its median by under "
            f"{worst_inst*100:.1f}% in either direction, which is where the bar "
            f"heights come from. The rest of each bar "
            f"is cProfile self-time apportioned over what is left after the "
            f"matrix products.",
            f"*{jax_note}")
        for line in textwrap.wrap(para, width=162))
    txt = fig.text(0.008, 0.012, caption, fontsize=7.6, color=INK_SOFT,
                   ha="left", va="bottom", linespacing=1.55)
    # Measure the caption instead of guessing a margin for it. Every earlier
    # version of this figure and its free-space sibling hardcoded the bottom of
    # the tight_layout rect, and every time a caption sentence was added the
    # x tick labels landed on top of it -- the caption grows by a line, the
    # constant does not. Rendering once and asking the text how tall it came out
    # makes the margin follow the words.
    fig.canvas.draw()
    top = txt.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height
    fig.tight_layout(rect=(0.004, top + 0.035, 0.992, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", help="internal: measure one adapter, print JSON")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved measurement rounds; the least "
                         "contaminated one is kept per adapter")
    ap.add_argument("--redraw", action="store_true",
                    help="redraw from the saved measurements, without measuring")
    ap.add_argument("--verbose", action="store_true",
                    help="print every profiled function with the bucket it was given")
    args = ap.parse_args()

    if args.measure:
        # Affinity before the heavy imports: thread pools size themselves from
        # the visible core count at import, not at call.
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
            except OSError:
                pass
        print(json.dumps(measure(args.measure, args.reps, args.verbose)))
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    from dragrace.case import Case
    case = Case.from_yaml(CASE).scan_cases()[0]

    print(f"measuring {CASE.name} at N_p={case.n_pupil}, N_f={case.n_focus}, "
          f"{args.reps} reps, one core")
    if args.redraw:
        saved = json.loads(DATA.read_text())
        saved["bars"] = [b for b in saved["bars"] if b["adapter"] in ADAPTERS]
        draw(saved["bars"], case.id, case.n_pupil, case.n_focus,
             saved["reps"], saved["rounds"], OUT)
        return 0

    bars = collect(args.reps, args.rounds, args.verbose)
    if not bars:
        print("no adapters produced a bar")
        return 1
    DATA.write_text(json.dumps(
        {"case": case.id, "config": CONFIG, "reps": args.reps,
         "rounds": args.rounds, "bars": bars}, indent=1))
    print(f"wrote {DATA}")
    draw(bars, case.id, case.n_pupil, case.n_focus, args.reps, args.rounds, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
