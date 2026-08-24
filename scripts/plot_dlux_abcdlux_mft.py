#!/usr/bin/env python
"""dLux against abcdLux on the pupil-to-focus board: what the kernel cache buys.

Same author, same language, same execution engine, same two GEMMs, same case,
same config, same core. One design difference:

    dLux      dLux.utils.propagation.MFT calls vmap(get_tf_mat)(shift) inside
              every propagation, so both (N_f, N_p) transfer matrices are
              re-exponentiated per call. It cannot hoist them: they depend on
              `wavelength`, which is the traced input of propagate_mono, and
              that is what buys polychromatic and wavelength-differentiable
              models.
    abcdLux   ships every propagator as a kernels()/kernel_prop() pair, so the
              caller holds (S, Kx, Ky) and hands them back. The adapter builds
              them in build() and calls fraunhofer_kernel_prop in propagate().

THE QUESTION THIS FIGURE HAD TO ANSWER HONESTLY. "abcdLux caches the MFT matrix,
so it is faster" is a hypothesis, not a caption, and drawing two curves with a
gap between them labelled "kernel cache" would assert it rather than test it.
Measured, the cache turns out to be most of the gap at N_p=128 and almost none
of it at N_p=2048, because the two costs it competes with grow at different
rates: rebuilding an (N_f, N_p) kernel is linear in N_p, and everything else
dLux redoes per call is quadratic.

HOW THE SPLIT IS MEASURED -- a ladder of AOT-compiled programs, each one rung
closer to dLux's own, all timed identically on one pinned core:

    A   abcdLux as the board runs it: kernels closed over, field the argument.
    B   A, then |.|^2, so both ends return the same quantity dLux does.
    C   dLux's own propagate_mono with the two transfer matrices replaced by
        constants computed at the case wavelength. Everything else is dLux's:
        initialise_wavefront, the Optic layer, the normalisation, the psf.
    D   dLux as the board runs it.

    D - C   rebuilding the MFT kernels
    C - B   rebuilding the pupil wavefront

The differences telescope, so the two components sum to the measured gap by
construction rather than by fit. What is NOT free is rung C being a faithful
variant of D, and that is checked rather than assumed: C's output is compared
against D's and must agree to ~1e-15 (printed, and shown on the figure). Note
that C is a measurement device, not a proposed implementation -- holding the
kernels constant while the pupil still follows a traced wavelength is not a
thing a dLux user could write, which is the whole point of the row above it.

Panel 1 is the board's own recorded medians (dragrace.report), gated and
measured under the harness's full protocol. Panel 2 is this script's ladder,
measured in one pinned subprocess. They are different measurements of the same
thing, so the script checks its endpoints against the recorded medians and
prints the disagreement instead of hiding it.

Pinned to one core with sched_setaffinity, the same mechanism and reason as
worker._pin_cpus: XLA honours no thread environment variable, so without
affinity both rows would be measured on a different machine from the board.

    python scripts/plot_dlux_abcdlux_mft.py [--reps 9]

Run it with the interpreter of the `dragrace` environment; the child inherits it.
Requires the board to have been measured first:

    dragrace run --case mft_array_scan --adapter abcdlux --config cpu_numpy_1t
    dragrace run --case mft_array_scan --adapter dlux    --config cpu_numpy_1t
    dragrace run --case mft_array_scan --adapter numpy_baseline --config cpu_numpy_1t
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CASE = "mft_array_scan"
CONFIG = "cpu_numpy_1t"
CASE_PATH = ROOT / "cases" / "pupil_to_focus" / "mft_array_scan.yaml"
CONFIG_PATH = ROOT / "configs" / "cpu_numpy_1t.yaml"
OUT = ROOT / "docs" / "figures" / "mft_dlux_vs_abcdlux.png"

#: dragrace.plots slots: dlux is 6 (#4a3aa7), abcdlux is 7 (#e34948). Taken from
#: the same table the board's own figures use, so a reader who has seen the
#: mft_array_scan curve finds the same two lines the same two colours here.
C_DLUX = "#4a3aa7"
C_ABCD = "#e34948"
C_KERNEL = "#eda100"          # the component the hypothesis named
#: Darker amber for text on the pale surface -- #eda100 is a fill colour and
#: fails contrast as 9pt type.
C_KERNEL_INK = "#8a5c00"
C_PUPIL = "#4a3aa7"           # dLux's own colour: this is dLux's model layer
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
CORRIDOR = "#b3b1ac"


# ------------------------------------------------------------------ child --
#: Runs in its own process for the reason every measurement here does: JAX fixes
#: x64 at first import and affinity has to be set before the thread pools exist.
CHILD = r'''
import json, os, statistics, sys, time

os.environ["JAX_ENABLE_X64"] = "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, sys.argv[1])
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})

REPS = int(sys.argv[2])
CASE_PATH, CONFIG_PATH, ADAPTERS_DIR = sys.argv[3], sys.argv[4], sys.argv[5]

import numpy as np
import jax, jax.numpy as jnp
import abcdLux
import dLux.utils as dlu
from dLux.utils.propagation import transfer_matrix, calc_nfringes

from dragrace import adapter as A
from dragrace.case import Case
from dragrace.config import Config

A.discover(ADAPTERS_DIR)
case = Case.from_yaml(CASE_PATH)
cfg = Config.from_yaml(CONFIG_PATH)


def median_s(fn, arg, reps):
    for _ in range(3):
        jax.block_until_ready(fn(arg))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


ab = A.get("abcdlux"); ab.configure(cfg)
dl = A.get("dlux");    dl.configure(cfg)

out = []
for c in case.scan_cases():
    sa = ab.build(c, cfg)
    sd = dl.build(c, cfg)

    # ---- rung B: abcdLux's propagation, then |.|^2, as one compiled program.
    S, Kx, Ky = abcdLux.fraunhofer_kernels(
        spec_in=(c.n_pupil, c.dx_pupil_m), spec_out=(c.n_focus, c.dx_focus_m),
        lam=c.wavelength_m, f=c.output.focal_length_m)
    fnB = jax.jit(
        lambda u: jnp.abs(abcdLux.fraunhofer_kernel_prop(u, S, Kx, Ky)) ** 2
    ).lower(sa["field"]).compile()

    # ---- rung C: dLux's propagate_mono with the transfer matrices hoisted.
    # Reconstructed from dLux's own call chain rather than reimplemented:
    # ParametricLayeredOpticalSystem.propagate_mono applies the layers and then
    # to_focus(), and AngularOpticalSystem.to_focus calls
    # Wavefront.propagate(psf_npixels, arcsec2rad(psf_pixel_scale/oversample)),
    # which is dlu.MFT. Only the two matrices and the normalisation constant are
    # replaced; every array-sized operation stays dLux's.
    optics = sd["optics"]
    wl = c.wavelength_m
    ps_in = optics.initialise_wavefront(wl, None).pixel_scale
    ps_out = dlu.arcsec2rad(optics.psf_pixel_scale / optics.oversample)
    n_out = optics.psf_npixels * optics.oversample
    # shift = 0 on both axes, so dLux's vmap over `shift` produces two identical
    # matrices; building one and reusing it is exact, not an approximation.
    tf = transfer_matrix(wl, c.n_pupil, ps_in, n_out, ps_out, 0.0, None, 0.0, False)
    nfr = calc_nfringes(wl, c.n_pupil, ps_in, n_out, ps_out, None)
    norm = jnp.exp(jnp.log(nfr) - (jnp.log(c.n_pupil) + jnp.log(n_out)))

    def hoisted(w, optics=optics, tf=tf, norm=norm):
        wf = optics.initialise_wavefront(w, None)
        for layer in list(optics.layers.values()):
            wf = layer(wf)
        return jnp.abs(((tf.T @ wf.phasor) @ tf) * norm) ** 2

    wlarg = sd["wavelength"]
    fnC = jax.jit(hoisted).lower(wlarg).compile()

    # The check that makes the ladder a decomposition rather than a story: rung
    # C must compute what dLux computes.
    oC = np.asarray(fnC(wlarg))
    oD = np.asarray(sd["fn"](wlarg))
    rel = float(np.linalg.norm(oC - oD) / np.linalg.norm(oD))

    # How much of each program is a full pupil-sized pass, straight from the
    # compiled HLO. Not used for timing -- it is the corroborating count for
    # what "rebuilding the pupil wavefront" means.
    def pupil_sized_ops(exe, n):
        import re, collections
        cnt = collections.Counter()
        for line in exe.as_text().splitlines():
            m = re.search(r"=\s+(\S+)\s+([a-z][a-z\-\.]*)\(", line)
            if m and f"{n},{n}" in m.group(1):
                cnt[m.group(2)] += 1
        return dict(cnt)

    out.append({
        "n": c.n_pupil,
        "A": median_s(sa["fn"], sa["field"], REPS),
        "B": median_s(fnB, sa["field"], REPS),
        "C": median_s(fnC, wlarg, REPS),
        "D": median_s(sd["fn"], wlarg, REPS),
        "rel_C_vs_D": rel,
        "flops_abcd": (sa.get("cost_analysis") or [{}])[0].get("flops")
                      if isinstance(sa.get("cost_analysis"), list)
                      else (sa.get("cost_analysis") or {}).get("flops"),
        "flops_dlux": (sd.get("cost_analysis") or [{}])[0].get("flops")
                      if isinstance(sd.get("cost_analysis"), list)
                      else (sd.get("cost_analysis") or {}).get("flops"),
        "bytes_abcd": (sa.get("cost_analysis") or [{}])[0].get("bytes accessed")
                      if isinstance(sa.get("cost_analysis"), list)
                      else (sa.get("cost_analysis") or {}).get("bytes accessed"),
        "bytes_dlux": (sd.get("cost_analysis") or [{}])[0].get("bytes accessed")
                      if isinstance(sd.get("cost_analysis"), list)
                      else (sd.get("cost_analysis") or {}).get("bytes accessed"),
        "ops_abcd": pupil_sized_ops(sa["fn"], c.n_pupil),
        "ops_dlux": pupil_sized_ops(sd["fn"], c.n_pupil),
    })
    ab.teardown(sa); dl.teardown(sd)

print("@@JSON@@" + json.dumps(out))
'''


def measure(reps: int) -> list[dict]:
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(ROOT / "src"), str(reps),
         str(CASE_PATH), str(CONFIG_PATH), str(ROOT / "adapters")],
        capture_output=True, text=True, env=env)
    tag = "@@JSON@@"
    for line in proc.stdout.splitlines():
        if line.startswith(tag):
            return json.loads(line[len(tag):])
    sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
    raise SystemExit("ladder subprocess produced no result")


# ------------------------------------------------------------ board rows --
def board() -> dict[str, dict[int, float]]:
    """Recorded medians for this case/config, through the harness's own filter.

    scan_rows applies the same three exclusions the timing table does -- status
    ok, gate pass, not traced -- so a point this figure draws is a point the
    report would print.
    """
    from dragrace.report import aggregate, scan_rows
    rows = scan_rows(aggregate(ROOT / "results"), case=CASE, config=CONFIG)
    out: dict[str, dict[int, float]] = {}
    machines = {r["machine"] for r in rows}
    if len(machines) > 1:
        # One figure per machine fingerprint, the same rule plots.py enforces.
        newest = max(rows, key=lambda r: r["utc"] or "")["machine"]
        rows = [r for r in rows if r["machine"] == newest]
    for r in rows:
        out.setdefault(r["adapter"], {})[int(r["scan_value"])] = r["median_s"]
    return out


# ------------------------------------------------------------------ draw --
def draw(rungs: list[dict], rec: dict[str, dict[int, float]], reps: int,
         path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ns = [r["n"] for r in rungs]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(15.4, 9.6), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # ---------------------------------------------------------- panel 1 ----
    ax.set_facecolor(SURFACE)
    base = rec.get("numpy_baseline", {})

    # Ideal-FLOP slope guide, normalised to the fastest code at the smallest
    # size -- the NumPy floor here. A slope to compare against, not a claim
    # about achievable time. Drawn first and thick so it reads as a guide.
    def ideal(n):
        return 8.0 * n * 128.0 * (n + 128.0)

    anchor = base or rec.get("abcdlux", {})
    if anchor:
        n0 = min(anchor)
        gy = [anchor[n0] * 1e3 * ideal(n) / ideal(n0) for n in ns]
        ax.plot(ns, gy, color=GRID, lw=7, zorder=0.5, solid_capstyle="round")
        ax.annotate("ideal FLOP scaling", xy=(ns[-1], gy[-1]), xytext=(9, -6),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=9.5, color=INK_SOFT)

    if base:
        bn = sorted(base)
        ln, = ax.plot(bn, [base[n] * 1e3 for n in bn], color=CORRIDOR, lw=1.7,
                      marker="s", ms=4.0, zorder=1.6)
        ln.set_dashes((5, 1.6))
        i = len(bn) // 2
        ax.annotate("NumPy floor\n(OpenBLAS zgemm, 1 core)",
                    xy=(bn[i], base[bn[i]] * 1e3), xytext=(6, -12),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=9.5, color=INK_SOFT, linespacing=1.35)

    for name, colour, label, marker, ms, dash in (
            ("dlux", C_DLUX, "dLux", "o", 6.5, (None, None)),
            ("abcdlux", C_ABCD, "abcdLux", "*", 11, (4, 1.2, 1, 1.2))):
        pts = rec.get(name, {})
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[n] * 1e3 for n in xs]
        ln, = ax.plot(xs, ys, color=colour, lw=2.1, marker=marker, ms=ms,
                      zorder=3, label=label)
        if dash[0] is not None:
            ln.set_dashes(dash)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=12.5,
                    color=colour, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(ns[0] * 0.84, ns[-1] * 1.75)
    ax.set_xlabel("pupil array size $N_p$ (samples across)", fontsize=11)
    ax.set_ylabel("propagation, median ms (log)", fontsize=11)
    ax.set_title("Runtime against array size", fontsize=13.5, color=INK,
                 loc="left", pad=10)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ax.spines.values():
        s_.set_color(GRID)

    # Ratio labels, placed to the LEFT of the geometric midpoint between the two
    # curves so they never sit on either line.
    dl_, ab_ = rec.get("dlux", {}), rec.get("abcdlux", {})
    for n in ns:
        if n in dl_ and n in ab_:
            ax.annotate(f"{dl_[n] / ab_[n]:.2f}x",
                        xy=(n * 1.055, (dl_[n] * ab_[n]) ** 0.5 * 1e3),
                        ha="left", va="center", fontsize=9.5, color=INK_SOFT,
                        bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE,
                                  ec=GRID, lw=0.6))

    # ---------------------------------------------------------- panel 2 ----
    # Drawn as curves rather than as stacked shares. The share is the headline
    # number and it is annotated on the amber points, but a stacked bar cannot
    # show WHY the share moves, and at N_p = 2048 the amber segment is 6% tall
    # and will not hold its own label. Two curves on log-log axes show the
    # mechanism directly: one slope is 1, the other is 2, and they cross.
    bx.set_facecolor(SURFACE)
    ker = [r["D"] - r["C"] for r in rungs]
    pup = [r["C"] - r["B"] for r in rungs]
    gap = [k + p for k, p in zip(ker, pup)]
    fk = [100.0 * k / g for k, g in zip(ker, gap)]

    # Slope guides anchored at each series' own first point: a line to compare
    # the measured slope against, not a fit through it.
    for exp_, anchor in ((1.0, ker[0]), (2.0, pup[0])):
        bx.plot(ns, [anchor * 1e3 * (n / ns[0]) ** exp_ for n in ns],
                color=GRID, lw=6, zorder=0.5, solid_capstyle="round")

    gl, = bx.plot(ns, [g * 1e3 for g in gap], color=INK_SOFT, lw=1.5, zorder=2)
    gl.set_dashes((5, 1.6))
    bx.plot(ns, [pp * 1e3 for pp in pup], color=C_PUPIL, lw=2.2, marker="s",
            ms=6.0, zorder=3)
    bx.plot(ns, [kk * 1e3 for kk in ker], color=C_KERNEL, lw=2.2, marker="o",
            ms=6.0, zorder=3)

    # The share the hypothesis is about, annotated on the series it is about.
    # (Lost once already to a careless splice, which is why the panel title
    # names it: a title promising a number the panel does not draw is worse
    # than no title.)
    for n, kk, f in zip(ns, ker, fk):
        bx.annotate(f"{f:.0f}%", xy=(n, kk * 1e3), xytext=(0, 13),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color=C_KERNEL_INK, fontweight="bold")

    # Where the two costs change places, interpolated in log-log from the
    # bracketing points rather than eyeballed off the figure or written down.
    import math
    cross = None
    for i in range(len(ns) - 1):
        if ker[i] > pup[i] >= 0 and ker[i + 1] <= pup[i + 1]:
            la = math.log(ker[i] / pup[i])
            lb = math.log(ker[i + 1] / pup[i + 1])
            t = la / (la - lb)
            cross = math.exp(math.log(ns[i]) + t * math.log(ns[i + 1] / ns[i]))
            break
    if cross:
        bx.axvline(cross, color=INK_SOFT, lw=0.9, ls=":", zorder=1)
        bx.annotate(f"they change places\nat $N_p \\approx$ {cross:.0f}",
                    xy=(cross, gap[-1] * 1e3), xytext=(7, -2),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=9.5, color=INK_SOFT, linespacing=1.35)

    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks(ns)
    bx.set_xticklabels([str(n) for n in ns])
    bx.set_xlim(ns[0] * 0.84, ns[-1] * 1.30)
    bx.set_xlabel("pupil array size $N_p$ (samples across)", fontsize=11)
    bx.set_ylabel("cost of one per-call rebuild, median ms (log)", fontsize=11)
    bx.set_title("What the gap is made of  \u2014  % is the kernel's share",
                 fontsize=13.5, color=INK, loc="left", pad=10)
    bx.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    bx.set_axisbelow(True)
    for s_ in bx.spines.values():
        s_.set_color(GRID)

    fig.legend(handles=[
        Line2D([], [], color=C_PUPIL, lw=2.2, marker="s", ms=6.0,
               label="pupil wavefront rebuilt per call  ($\\propto N_p^2$)"),
        Line2D([], [], color=INK_SOFT, lw=1.5, ls=(0, (5, 1.6)),
               label="total gap (dLux \u2212 abcdLux)"),
        Line2D([], [], color=C_KERNEL, lw=2.2, marker="o", ms=6.0,
               label="MFT transfer matrices rebuilt per call  ($\\propto N_p$)"),
        Line2D([], [], color=GRID, lw=6,
               label="$N_p$ and $N_p^2$ slope guides")],
        loc="upper left", bbox_to_anchor=(0.435, 0.335), ncol=2,
        frameon=False, fontsize=10, columnspacing=2.2)

    # ---------------------------------------------------------- caption ----
    worst_rel = max(r["rel_C_vs_D"] for r in rungs)
    big = rungs[-1]
    n_ops_d = sum(big["ops_dlux"].values())
    n_ops_a = sum(big["ops_abcd"].values())
    floor_ratio = (rec["abcdlux"][ns[-1]] / base[ns[-1]]) if base else float("nan")
    #: Worst disagreement between the ladder's own endpoints and the medians the
    #: board recorded. Computed, never asserted: it is the one number that says
    #: whether the two measurements describe the same machine state.
    worst_end = 100.0 * max(
        abs(r[k] / rec[a][r["n"]] - 1.0)
        for r in rungs for k, a in (("A", "abcdlux"), ("D", "dlux"))
        if rec.get(a, {}).get(r["n"]))
    para = (
        f"mft_array_scan, cpu_numpy_1t, complex128, one pinned core. N_f = 128 is "
        f"fixed at every point, so N_p is the only axis moving. LEFT: the board's "
        f"own recorded medians -- 25 repeats after 3 warm-ups, gated at 1e-10, and "
        f"both codes land at 3e-15 or better, so the gap is cost and not accuracy. "
        f"abcdLux runs parallel to ideal scaling at {floor_ratio:.2f}x the NumPy "
        f"floor: that offset is XLA's complex128 GEMM against OpenBLAS's, a "
        f"property of the engine rather than of either library.  RIGHT: that gap, "
        f"split by a ladder of compiled programs ({reps} repeats each, one process, "
        f"same core). The parts telescope, so they sum to the measured gap by "
        f"construction rather than by fit, and the rung holding dLux's transfer "
        f"matrices constant reproduces dLux's own output to {worst_rel:.0e} -- which "
        f"is what makes this a decomposition and not a story. Its endpoints "
        f"reproduce the recorded medians to within {worst_end:.0f}% at every size. "
        f" THE HYPOTHESIS TESTED -- 'abcdLux is "
        f"faster because it caches the MFT matrix' -- holds at N_p = {ns[0]} "
        f"({fk[0]:.0f}% of the gap) and fails at N_p = {ns[-1]} ({fk[-1]:.0f}%): an "
        f"(N_f, N_p) kernel rebuild grows linearly in N_p, a pupil rebuild "
        f"quadratically -- the grey bands are those two slopes anchored at the "
        f"smallest size -- and the second overtakes the first at N_p \u2248 "
        f"{cross:.0f}. "
        f"That second cost is dLux's model layer rather than a defect -- "
        f"propagate_mono takes a wavelength, so the transmission, the OPD-to-phase "
        f"conversion and the phasor all sit downstream of a traced input. Its "
        f"compiled program carries {n_ops_d} instructions at the full "
        f"{big['n']}x{big['n']} pupil size against abcdLux's {n_ops_a} (the input "
        f"parameter), touching {big['bytes_dlux'] / big['bytes_abcd']:.1f}x the "
        f"memory for {big['flops_dlux'] / big['flops_abcd']:.2f}x the arithmetic. "
        f"abcdLux buys neither polychromatic sources nor wavelength gradients, and "
        f"charges for neither."
    )
    fig.text(0.012, 0.265, "\n".join(textwrap.wrap(para, width=178)),
             ha="left", va="top", fontsize=8.9, color=INK_SOFT, linespacing=1.62)

    fig.suptitle("dLux and abcdLux on the same MFT: what caching the kernel is worth",
                 fontsize=16.5, color=INK, x=0.012, ha="left", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.972, top=0.895, bottom=0.415,
                        wspace=0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    rec = board()
    missing = [a for a in ("dlux", "abcdlux") if not rec.get(a)]
    if missing:
        raise SystemExit(
            f"no recorded {CASE} / {CONFIG} points for {missing}. Run them first "
            f"-- see this module's docstring.")

    rungs = measure(args.reps)

    print(f"{'N':>6} {'A abcd':>9} {'B +|.|2':>9} {'C hoist':>9} {'D dlux':>9} "
          f"{'ker':>8} {'pupil':>9} {'rel C~D':>9}")
    for r in rungs:
        print(f"{r['n']:6d} {r['A']*1e3:9.3f} {r['B']*1e3:9.3f} {r['C']*1e3:9.3f} "
              f"{r['D']*1e3:9.3f} {(r['D']-r['C'])*1e3:8.3f} "
              f"{(r['C']-r['B'])*1e3:9.3f} {r['rel_C_vs_D']:9.1e}")

    # The endpoints of the ladder are the two things the board also measured.
    # Disagreement is reported rather than absorbed: it is the only signal that
    # the ladder ran on a different machine state from the recorded curve.
    print("\nladder endpoints vs recorded medians (ladder/recorded):")
    for r in rungs:
        a = rec["abcdlux"].get(r["n"])
        d = rec["dlux"].get(r["n"])
        if a and d:
            print(f"  N={r['n']:5d}  abcdLux {r['A']/a:5.3f}   dLux {r['D']/d:5.3f}")

    draw(rungs, rec, args.reps, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
