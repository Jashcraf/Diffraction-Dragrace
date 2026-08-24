#!/usr/bin/env python
"""Where one free-space propagation actually spends its time, per code.

The scan curve on the plane_to_plane board says POPPY and PROPER cost 2.2x and
2.5x the NumPy floor at N=4096, and says nothing about why. This figure answers
that: one stacked bar per code, split into

    FFT             the two transforms -- the only work the physics requires
    transfer fn     rebuilding the free-space kernel, per call
    array shuffling whole-array copies that move no physics (fftshift by roll,
                    .copy(), np.ones scratch)
    other           everything else: normalisation passes, object construction,
                    unit checks, PROPER's gc.collect()

Every code here computes the SAME quantity to the same tolerance -- the board
gates all six at 1e-10 and they land at 1e-13 or better. So the bars are not a
quality trade. They are the cost of what each library does around a transform
it does not do faster than anyone else.

HOW THE SPLIT IS MEASURED. cProfile, bucketing tottime (self time, children
excluded) by an explicit rule table -- RULES below, first match wins. Bucket
FRACTIONS come from the profile; the bar HEIGHT is the median measured with no
profiler attached, and the fractions are rescaled onto it. Profiling inflates
a run by a few percent and it inflates the Python-heavy adapters most, which is
exactly the axis this figure compares -- so the profiler is allowed to say where
the time went and never how much there was.

Two adapters cannot be read that way, and both are handled explicitly rather
than quietly:

  PROPER   prop_ptp builds the kernel AND applies five whole-array rescaling
           passes in one function body, so cProfile attributes both to the same
           tottime. `_split_proper` measures the two groups standalone -- the
           lines copied verbatim from prop_ptp -- and divides that tottime in
           the measured ratio.
  dLux     is one fused XLA program; cProfile sees only block_until_ready. Its
           split is bracketed instead: `fft` is the same c128 transform pair in
           XLA with the kernel held as a constant, `transfer fn` is the
           increment when the kernel is instead built from the traced
           wavelength -- which is what dLux's own HLO does, see
           %complex_multiply_fusion -- and the rest is the remainder. The bar
           is hatched to say so.

Pinned to one core with sched_setaffinity, the same mechanism and for the same
reason as worker._pin_cpus: XLA honours no thread environment variable, so
without affinity dLux alone would run on every core and its bar would be a
comparison against a different machine.

Each adapter is measured in its own subprocess. They mutate global state to
configure themselves -- prysm repoints mathops, HCIPy rewrites its
Configuration, JAX fixes x64 at first import -- and in one process the last one
configured would silently set the terms for the rest.

    python scripts/plot_free_space_breakdown.py [--n 2048] [--reps 8]

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
CASE = ROOT / "cases" / "plane_to_plane" / "fresnel_scan_z1m.yaml"
CONFIG = "cpu_numpy_1t"
OUT = ROOT / "docs" / "figures" / "fresnel_free_space_breakdown.png"

#: Left to right. Fixed rather than sorted by runtime, so the figure keeps its
#: shape when a code is added or drops off the board.
ADAPTERS = ("numpy_baseline", "hcipy", "prysm", "dlux", "poppy", "proper")

BUCKETS = ("fft", "kernel", "copies", "other")
LABELS = {
    "fft": "FFT \u2014 the two transforms, the only required work",
    "kernel": "transfer function, rebuilt every call",
    "copies": "array shuffling: fftshift-by-copy, scratch, casts",
    "other": "other per-call overhead",
}
#: Repo categorical palette (dragrace.plots.PALETTE) for the three named
#: buckets; the residual is deliberately inert grey so it never reads as a
#: finding. Every segment is also value-labelled, so nothing here rests on hue.
COLORS = {"fft": "#2a78d6", "kernel": "#eda100",
          "copies": "#e34948", "other": "#b3b1ac"}

INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

# --------------------------------------------------------------- bucketing --
#: (substring of "file:lineno(function)", bucket). First match wins, so the
#: specific rules come before the general ones. Anything unmatched is `other`,
#: and --verbose prints every function above the noise floor with the bucket it
#: was given, so a wrong rule shows up as a wrong label rather than as a
#: plausible bar.
RULES = [
    # the transform itself
    ("_pocketfft.py", "fft"),
    ("pocketfft", "fft"),
    # per-call construction of the free-space kernel, and the coordinate grids
    # that exist only to feed it
    ("necompiler.py", "kernel"),                      # POPPY: numexpr exp+coords
    ("fresnel.py:680(_propagate_ptp)", "kernel"),     # POPPY: rho^2 arithmetic
    ("(coordinates)", "kernel"),
    ("(indices)", "kernel"),
    ("angular_spectrum_transfer_function", "kernel"),  # prysm
    ("prop_ptp.py", "kernel"),                        # PROPER: see _split_proper
    ("(outer)", "kernel"),
    ("fftfreq", "kernel"),
    ("_shape_base_impl.py", "kernel"),                # np.tile
    # whole-array copies that move no physics
    ("(roll)", "copies"),
    ("_helper.py", "copies"),                         # fftshift / ifftshift
    ("method 'copy' of 'numpy", "copies"),
    ("built-in method numpy.array", "copies"),
    ("method 'astype'", "copies"),
    ("(ones)", "copies"),
    ("(zeros)", "copies"),
    ("method 'repeat'", "copies"),
    ("prop_shift_center", "copies"),
    ("pad_to_oversample", "copies"),
    ("prop_multiply", "copies"),
]


#: Per-call FFT timings for the region currently being measured. The FFT bucket
#: does NOT come from the profile any more, and the reason is a wrong figure.
#:
#: It used to be `fft_tottime / profiled_total * unprofiled_median`, i.e. a
#: fraction of one measurement rescaled onto another. Both factors carry the
#: run-to-run variance of a 2048^2 out-of-cache transform, which is large: three
#: interleaved rounds of the identical two calls measured 132-152 ms for EVERY
#: code here, and the ranking between codes reordered completely from round to
#: round. The published figure drew PROPER's transforms at 132 ms and prysm's at
#: 152 and invited the reading that PROPER has a cheaper FFT. It does not --
#: measured directly, all five NumPy-backed codes make exactly two
#: np.fft calls on a (2048, 2048) C-contiguous complex128 array with identical
#: alignment, and no code owns a position in that band.
#:
#: So the transform is now TIMED, not attributed: the entry points are wrapped
#: before any propagator is imported, and cProfile is left to apportion only the
#: remainder, where the differences are real and an order of magnitude larger
#: than this noise.
_FFT_CALLS: list[float] = []


def _install_fft_timers() -> None:
    """Wrap NumPy/SciPy FFT entry points with a stopwatch.

    Must run before the propagators are imported: several of them bind
    `np.fft.*` into closures at import (hcipy/_math/fft.py captures
    `getattr(np.fft, name)` at module scope), so a library imported first would
    hold the unwrapped function and report zero. Same ordering constraint, and
    the same reason, as dragrace.flops.ledger.record().
    """
    import numpy as np

    def wrap(mod, attr):
        fn = getattr(mod, attr)

        def timed(*args, **kwargs):
            t = time.perf_counter()
            out = fn(*args, **kwargs)
            _FFT_CALLS.append(time.perf_counter() - t)
            return out

        timed.__name__ = attr
        setattr(mod, attr, timed)

    for a in ("fft2", "ifft2", "fftn", "ifftn", "fft", "ifft", "rfft2", "irfft2"):
        if hasattr(np.fft, a):
            wrap(np.fft, a)
    try:
        import scipy.fft as sfft
    except ImportError:
        return
    for a in ("fft2", "ifft2", "fftn", "ifftn"):
        if hasattr(sfft, a):
            wrap(sfft, a)


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


def _split_proper(n: int, case) -> float:
    """Fraction of prop_ptp's self-time that is kernel construction.

    prop_ptp builds the kernel and rescales the whole array five times in one
    function body, so cProfile cannot separate them. These are its lines,
    verbatim, timed as two groups. The rolls are excluded here because cProfile
    already charges them to `copies` as their own frames.
    """
    import numpy as np

    samp = case.dx_pupil_m
    lam, dz = case.wavelength_m, case.propagation.distance_m
    i = np.array([0 + 1j], dtype=np.complex128)
    rng = np.random.default_rng(0)
    wfarr = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    def kernel():
        xr = np.tile(((np.arange(n, dtype=np.float64) - int(n / 2)) / (n * samp)) ** 2, (n, 1))
        rh = xr + np.transpose(xr)
        return np.exp((-i * np.pi * lam * dz) * rh)

    exp_t = kernel()

    def rescale():
        w = wfarr
        w /= np.size(w)
        w *= n
        w *= exp_t
        w *= np.size(w)
        w /= n

    tk, tr = _median(kernel, 7), _median(rescale, 7)
    return tk / (tk + tr)


def measure(name: str, n: int, reps: int, verbose: bool) -> dict:
    """Run one adapter and return its bar. Called in the child process."""
    import cProfile
    import pstats

    _install_fft_timers()          # BEFORE any propagator is imported
    sys.path.insert(0, str(ROOT / "src"))
    from dragrace.case import Case
    from dragrace.config import load_configs
    import dragrace.adapter as adapters

    adapters.discover(str(ROOT / "adapters"))
    case = Case.from_yaml(CASE)
    config = load_configs(str(ROOT / "configs"))[CONFIG]
    sub = next(s for s in case.scan_cases() if s.n_pupil == n)

    ad = adapters.get(name)
    ok = ad.configure(config)
    if ok is not True:
        return {"adapter": name, "skipped": str(ok)}
    state = ad.build(sub, config)

    def one():
        result = ad.propagate(state)
        ad.sync(result)

    if name == "dlux":
        return _measure_dlux(ad, state, n, reps, one)

    # The transforms, timed per call in the same unprofiled region that
    # produces the bar height -- so the FFT segment is a measurement rather
    # than a share of one.
    import numpy as np

    per_rep = []
    for _ in range(3):
        one()
    for _ in range(reps):
        _FFT_CALLS.clear()
        one()
        per_rep.append((sum(_FFT_CALLS), len(_FFT_CALLS)))
    median = _median(one, reps)
    fft_s = float(np.median([r[0] for r in per_rep]))
    n_fft = per_rep[-1][1]

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
        if tt / reps > 0.0008:
            rows.append((tt / reps, b, nc / reps, key.split("site-packages/")[-1]))

    if name == "proper" and buckets["kernel"] > 0:
        frac = _split_proper(n, sub)
        moved = buckets["kernel"] * (1.0 - frac)
        buckets["kernel"] -= moved
        buckets["other"] += moved

    # The profile now apportions only what is left after the transforms. Its
    # own fft tottime is dropped rather than reused: it is the same quantity,
    # measured worse.
    rest = {k: v for k, v in buckets.items() if k != "fft"}
    rest_total = sum(rest.values()) or 1.0
    room = max(median - fft_s, 0.0)
    out = {"adapter": name, "n": n, "median_s": median, "n_fft_calls": n_fft,
           "fft_measured": True,
           "buckets_s": dict({"fft": fft_s},
                             **{k: v / rest_total * room for k, v in rest.items()})}
    if verbose:
        out["rows"] = sorted(rows, reverse=True)[:12]
    return out


def _measure_dlux(ad, state, n: int, reps: int, one) -> dict:
    """dLux's bracket. See the module docstring for why it is not a profile."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    rho = jnp.asarray(rng.standard_normal((n, n)) ** 2)
    tf = jnp.exp(-1j * np.pi * 632.8e-9 * rho)
    wl = jnp.asarray(632.8e-9)

    const_kernel = jax.jit(lambda a: jnp.fft.ifft2(jnp.fft.fft2(a) * tf))
    built_kernel = jax.jit(
        lambda a, w: jnp.fft.ifft2(jnp.fft.fft2(a) * jnp.exp(-1j * jnp.pi * w * rho)))
    const_kernel(x).block_until_ready()
    built_kernel(x, wl).block_until_ready()

    median = _median(one, reps)
    t_fft = _median(lambda: const_kernel(x).block_until_ready(), reps)
    t_both = _median(lambda: built_kernel(x, wl).block_until_ready(), reps)
    kernel = max(t_both - t_fft, 0.0)
    return {"adapter": "dlux", "n": n, "median_s": median, "bracketed": True,
            "buckets_s": {"fft": t_fft, "kernel": kernel, "copies": 0.0,
                          "other": max(median - t_fft - kernel, 0.0)}}


# ------------------------------------------------------------------ driver --
def _one_round(name: str, n: int, reps: int, verbose: bool) -> dict | None:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               JAX_ENABLE_X64="1")              # the case is complex128
    cmd = [sys.executable, __file__, "--measure", name,
           "--n", str(n), "--reps", str(reps)]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = next((ln for ln in reversed(proc.stdout.splitlines())
                 if ln.startswith("{")), None)
    if line is None:
        print(f"  {name}: no result\n{proc.stdout[-800:]}{proc.stderr[-800:]}")
        return None
    return json.loads(line)


def collect(n: int, reps: int, rounds: int, verbose: bool) -> list[dict]:
    """Measure every adapter `rounds` times and keep its least contaminated run.

    A 2048^2 complex128 transform is far out of cache, and its cost moves by
    more than 10% between processes on an otherwise idle machine -- more than
    the differences between the five codes that all make the identical two
    calls. One round per adapter therefore ranks noise. Rounds are interleaved
    rather than repeated per adapter, so a slow patch of machine time cannot
    land entirely on one bar; the round with the smallest total is kept whole,
    because noise adds time and never removes it (the same argument
    report.best_points makes for a re-measured scan point), and taking each
    bucket's minimum separately would mix rounds and could leave the segments
    not summing to any measured total.
    """
    best: dict[str, dict] = {}
    spread: dict[str, list[float]] = {}
    for r in range(rounds):
        for name in ADAPTERS:
            bar = _one_round(name, n, reps, verbose and r == 0)
            if bar is None or bar.get("skipped"):
                if bar and bar.get("skipped"):
                    print(f"  {name}: skipped -- {bar['skipped']}")
                continue
            spread.setdefault(name, []).append(bar["buckets_s"]["fft"])
            cur = best.get(name)
            if cur is None or bar["median_s"] < cur["median_s"]:
                best[name] = bar
        print(f"  -- round {r + 1}/{rounds} done")

    bars = []
    for name in ADAPTERS:
        if name not in best:
            continue
        bar = best[name]
        # Carried onto the figure: the span of this code's OWN transform time
        # across rounds is what says whether the differences between codes mean
        # anything, and it must not be something the reader has to take on faith.
        bar["fft_span_s"] = [min(spread[name]), max(spread[name])]
        bars.append(bar)
        b = bar["buckets_s"]
        lo, hi = bar["fft_span_s"]
        print(f"  {bar['adapter']:15s} {bar['median_s']*1e3:8.2f} ms   " +
              "  ".join(f"{k} {b[k]*1e3:7.2f}" for k in BUCKETS) +
              f"   fft over rounds {lo*1e3:6.1f}-{hi*1e3:6.1f}")
        for row in bar.get("rows", []):
            print(f"      {row[0]*1e3:8.2f} ms {row[2]:5.1f}x [{row[1]:6s}] {row[3][:88]}")
    return bars


def draw(bars: list[dict], n: int, reps: int, rounds: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(11.6, 8.2), dpi=190)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

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
            if h > 0.055 * max(b["median_s"] for b in bars) * 1e3:
                ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", zorder=4,
                        # the residual bucket is a pale grey; white on it is
                        # unreadable, and this is the one label that matters
                        # most on dLux's bar
                        color=INK if key == "other" else "white")
            bottom += h
        ax.text(x, bottom + 6, f"{bar['median_s']*1e3:.0f} ms", ha="center",
                va="bottom", fontsize=10, color=INK, fontweight="bold", zorder=4)

    # A BAND, not a floor line. The five NumPy-backed codes make the identical
    # two calls -- same shape, dtype, contiguity and alignment -- and the band
    # spans every transform time measured across every round. Drawing a line at
    # the smallest of them, as this figure once did, states an ordering the
    # measurement does not support: over three rounds the ranking reordered
    # completely, and PROPER held both the fastest and the slowest position.
    # Bounded by the per-code BEST across rounds, which is exactly the value
    # each bar's blue segment carries -- so no bar can sit outside the band that
    # is telling the reader not to rank the bars. The run-to-run instability is
    # a number in the caption instead of a shape here: one contended round put
    # HCIPy's pair at 197 ms, and a band stretched to that would hide the
    # segments it sits behind while saying less than the sentence does.
    nb = [b for b in bars if not b.get("bracketed")]
    lo = min(b["buckets_s"]["fft"] for b in nb) * 1e3
    hi = max(b["buckets_s"]["fft"] for b in nb) * 1e3
    ax.axhspan(lo, hi, color=INK_SOFT, alpha=0.13, lw=0, zorder=2)
    ax.axhline(lo, color=INK_SOFT, lw=0.8, ls=(0, (5, 3)), zorder=2)
    ax.axhline(hi, color=INK_SOFT, lw=0.8, ls=(0, (5, 3)), zorder=2)
    # Annotated in a right-hand margin rather than over the bars: every band
    # inside the axes is occupied at the height this sits at, and a label laid
    # over a segment hides the number it is meant to explain.
    ax.set_xlim(-0.62, len(bars) - 1 + 0.62 + 1.30)
    ax.text(len(bars) - 1 + 0.50, hi + 6,
            f"  the two transforms:\n  {lo:.0f}-{hi:.0f} ms across all five\n"
            f"  NumPy-backed codes.\n  Their order inside this\n  band is not resolved.",
            ha="left", va="bottom", fontsize=8.5, color=INK_SOFT, style="italic",
            linespacing=1.45)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([b["adapter"] + ("*" if b.get("bracketed") else "")
                        for b in bars], fontsize=10.5, color=INK)
    ax.set_ylabel("runtime per propagation  (ms, one core)", fontsize=10.5, color=INK)
    ax.set_title(f"One paraxial free-space propagation, {n}x{n} complex128 — "
                 f"where the time goes",
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

    # Wrapped, not trusted to fit: an unwrapped caption runs off the canvas and
    # the part that falls off is always the caveat at the end.
    #: Worst round-to-round swing any single code showed in its OWN transform
    #: pair. This is the number that decides whether the ordering inside the
    #: band means anything, so it is measured, not asserted.
    swing = max((max(b["fft_span_s"]) - min(b["fft_span_s"])) * 1e3
                for b in bars if not b.get("bracketed") and b.get("fft_span_s"))
    caption = "\n".join(
        line for para in (
            f"fresnel_scan_z1m at N={n}, {CONFIG}, complex128, one core "
            f"(sched_setaffinity, as worker._pin_cpus does). Median of {reps} "
            f"calls. All six agree with the reference kernel to 1e-13 or better, "
            f"so this is cost and not accuracy.",
            f"Bar heights are the smallest of {rounds} interleaved rounds, "
            f"measured with no profiler attached. THE BLUE SEGMENT IS TIMED, NOT "
            f"ATTRIBUTED: the np.fft entry points are wrapped before any "
            f"propagator is imported, so the transform is a measurement rather "
            f"than a share of one. It has to be -- the five NumPy-backed codes "
            f"make the identical two calls on a ({n}, {n}) C-contiguous "
            f"complex128 array, and a single code's own pair moved by as much "
            f"as {swing:.0f} ms between rounds on an idle machine -- more than "
            f"the whole spread between codes, so their order inside the grey "
            f"band is noise and not a property of any library. (An earlier "
            f"version of this figure derived the segment from the profile "
            f"instead and drew PROPER's transforms as the cheapest here; "
            f"repeated, that ordering does not survive.) The rest of each bar "
            f"is cProfile self-time apportioned over what is left after the "
            f"transforms. "
            f"*dLux is one fused XLA program and cannot be profiled that way: "
            f"its split is bracketed by timing the same transform pair in XLA "
            f"with the kernel held constant, then rebuilt from the traced "
            f"wavelength. Its FFT is XLA's, not NumPy's, so its bottom segment "
            f"is not comparable with the others'.")
        for line in textwrap.wrap(para, width=158))
    fig.text(0.008, 0.012, caption, fontsize=7.6, color=INK_SOFT,
             ha="left", va="bottom", linespacing=1.55)
    fig.tight_layout(rect=(0.004, 0.205, 0.992, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", help="internal: measure one adapter, print JSON")
    ap.add_argument("--n", type=int, default=2048, help="scan point (array size)")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved measurement rounds; the least "
                         "contaminated one is kept per adapter")
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
        print(json.dumps(measure(args.measure, args.n, args.reps, args.verbose)))
        return 0

    print(f"measuring {CASE.name} at N={args.n}, {args.reps} reps, one core")
    bars = collect(args.n, args.reps, args.rounds, args.verbose)
    if not bars:
        print("no adapters produced a bar")
        return 1
    draw(bars, args.n, args.reps, args.rounds, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
