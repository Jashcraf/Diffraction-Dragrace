"""Aggregate the per-package JSON results into comparison plots + a table.

Reads every ``results/*.json`` produced by the benchmark scripts and emits:

    results/comparison_A_nograd.png     time vs N, all 4 packages, no back-prop
    results/comparison_B_backprop.png   time vs N, prysm & dLux, back-prop vs not
                                        (with 1-sigma error bars over the trials)
    results/summary_bar.png             grouped bars at the largest common N
    results/summary.md                  a Markdown table of every run

Runs in any env with numpy + matplotlib (e.g. the base env)::

    python plot_results.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Stable colour + marker per package so the two figures read consistently.
STYLE = {
    "POPPY": ("#d1495b", "o"),
    "HCIPy": ("#edae49", "s"),
    "prysm": ("#00798c", "^"),
    "dLux":  ("#30638e", "D"),
    "PROPER": ("#8f4d9e", "P"),   # forward-model-only comparison
}


def load(subdir=""):
    records = []
    for path in sorted(glob.glob(os.path.join(RESULTS, subdir, "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    return records


# The seven memory-footprint lines: (label, colour, linestyle, marker, predicate).
def _is(pkg, mode, jit=None):
    return lambda r: (r["package"] == pkg and r["mode"] == mode
                      and (jit is None or r.get("jit") == jit))


MEM_CASES = [
    ("POPPY",                      "#d1495b", "-",  "o", _is("POPPY", "nograd")),
    ("HCIPy",                      "#edae49", "-",  "s", _is("HCIPy", "nograd")),
    ("prysm (finite diff)",        "#00798c", "-",  "^", _is("prysm", "nograd")),
    ("prysm (back-prop)",          "#00798c", "--", "v", _is("prysm", "backprop")),
    ("dLux (finite diff, no jit)", "#30638e", ":",  "D", _is("dLux", "nograd", False)),
    ("dLux (back-prop, no jit)",   "#30638e", "--", "P", _is("dLux", "backprop", False)),
    ("dLux (back-prop, jit)",      "#30638e", "-",  "X", _is("dLux", "backprop", True)),
]


def plot_memory(records):
    """Peak-RSS footprint vs N for the seven requested cases (Comparison D)."""
    if not records:
        print("No memory results in results/mem/ - run run_mem.py first.")
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, color, ls, marker, pred in MEM_CASES:
        rows = sorted((r for r in records if pred(r)
                       and "mem_footprint_mb" in r), key=lambda r: r["n"])
        if not rows:
            continue
        ns = [r["n"] for r in rows]
        mem = [r["mem_footprint_mb"] for r in rows]
        ax.plot(ns, mem, ls=ls, marker=marker, color=color, lw=2, ms=7,
                label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("pupil pixels across (N)")
    ax.set_ylabel("peak memory footprint [MiB]  (RSS above import baseline)")
    ax.set_title("Comparison D — memory footprint of phase retrieval\n"
                 "Subaru-like asymmetric pupil, N = 2^6 .. 2^10")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "memory_footprint.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def _trial_std_ms(record):
    """Sample standard deviation (ddof=1) of the timed trials, in ms.

    Returns 0.0 when a record predates ``time_all_s`` or holds a single trial,
    so an old result plots as a bare marker rather than crashing.
    """
    times = record.get("time_all_s") or []
    if len(times) < 2:
        return 0.0
    return float(np.std(times, ddof=1)) * 1e3


def _series(records, package, mode, jit="nofalse"):
    """(sorted Ns, median times ms, trial std ms) for one package+mode.

    ``jit`` selects the dLux variant: ``True`` = jitted, ``False`` = eager, or
    the default ``"nofalse"`` which keeps everything except eager (used for the
    numpy packages, whose records have no ``jit`` field).
    """
    def keep(r):
        if r["package"] != package or r["mode"] != mode:
            return False
        if jit == "nofalse":
            return r.get("jit") is not False
        return r.get("jit") == jit
    rows = sorted((r for r in records if keep(r)), key=lambda r: r["n"])
    return ([r["n"] for r in rows],
            [r["time_median_s"] * 1e3 for r in rows],
            [_trial_std_ms(r) for r in rows])


def _plot_dlux_pair(ax, records, mode, prefix, yerr=False):
    """Draw dLux jit (solid ◆) and no-jit (dashed ▽) lines for a given mode.

    ``yerr=True`` adds 1-sigma error bars over the timed trials (Comparison B).
    """
    c = STYLE["dLux"][0]
    for jit, marker, ls, ms, tag in ((True, "D", "-", 8, "jit"),
                                     (False, "v", "--", 7, "no jit")):
        ns, t, e = _series(records, "dLux", mode, jit=jit)
        if not ns:
            continue
        label = f"dLux{prefix} ({tag})"
        if yerr:
            ax.errorbar(ns, t, yerr=e, marker=marker, color=c, lw=2, ms=ms,
                        ls=ls, capsize=4, label=label)
        else:
            ax.plot(ns, t, marker=marker, color=c, lw=2, ms=ms, ls=ls,
                    label=label)


def plot_comparison_a(records):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pkg in ["POPPY", "HCIPy", "prysm"]:
        ns, t, _ = _series(records, pkg, "nograd")
        if not ns:
            continue
        c, m = STYLE[pkg]
        ax.plot(ns, t, marker=m, color=c, label=pkg, lw=2, ms=8)
    _plot_dlux_pair(ax, records, "nograd", "")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("pupil pixels across (N)")
    ax.set_ylabel("phase-retrieval wall time [ms]  (median)")
    ax.set_title("Comparison A — no gradient back-prop (finite-difference L-BFGS-B)\n"
                 "Subaru-like asymmetric pupil; dLux shown jit vs no-jit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="propagator", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "comparison_A_nograd.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def plot_comparison_b(records):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cp, mp = STYLE["prysm"]
    ns, t, e = _series(records, "prysm", "backprop")
    if ns:
        ax.errorbar(ns, t, yerr=e, marker=mp, color=cp, lw=2, ms=8, capsize=4,
                    label="prysm (back-prop)")
    _plot_dlux_pair(ax, records, "backprop", " (back-prop)", yerr=True)
    # Trial count is uniform across a run; report it in the caption.
    n_trials = {r["trials"] for r in records if r["mode"] == "backprop"}
    trial_txt = (f"{n_trials.pop()}" if len(n_trials) == 1
                 else "/".join(str(x) for x in sorted(n_trials)))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("pupil pixels across (N)")
    ax.set_ylabel("phase-retrieval wall time [ms]  (median)")
    ax.set_title("Comparison B — gradient back-prop (L-BFGS-B)\n"
                 f"analytic/AD gradient; dLux shown jit vs no-jit\n"
                 f"error bars = 1$\\sigma$ over {trial_txt} trials")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "comparison_B_backprop.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def plot_jit(records):
    """dLux back-prop: JIT-compiled vs eager, time vs N."""
    def dlux_series(jit_flag):
        rows = [r for r in records if r["package"] == "dLux"
                and r["mode"] == "backprop" and r.get("jit") == jit_flag]
        rows.sort(key=lambda r: r["n"])
        return [r["n"] for r in rows], [r["time_median_s"] * 1e3 for r in rows]

    ns_j, t_j = dlux_series(True)
    ns_e, t_e = dlux_series(False)
    if not ns_j and not ns_e:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    c = STYLE["dLux"][0]
    if ns_j:
        ax.plot(ns_j, t_j, marker="D", color=c, lw=2, ms=8, label="jax.jit")
    if ns_e:
        ax.plot(ns_e, t_e, marker="o", color="#888", lw=2, ms=8, ls="--",
                label="eager (no jit)")
    # Annotate the speedup where both exist.
    common_ns = sorted(set(ns_j) & set(ns_e))
    for n in common_ns:
        tj = t_j[ns_j.index(n)]; te = t_e[ns_e.index(n)]
        ax.annotate(f"{te / tj:.1f}x", (n, tj), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=c)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("pupil pixels across (N)")
    ax.set_ylabel("phase-retrieval wall time [ms]  (median)")
    ax.set_title("dLux back-prop: JIT vs eager\n"
                 "(compilation excluded via warm-up; label = eager/jit speedup)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS, "dlux_jit_vs_eager.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


# (label, colour, linestyle, marker, package, mode, jit-selector)
THRU_CASES = [
    ("POPPY (loop)",         "#d1495b", "-",  "o", "POPPY", "loop", None),
    ("HCIPy (loop)",         "#edae49", "-",  "s", "HCIPy", "loop", None),
    ("prysm (loop)",         "#00798c", "-",  "^", "prysm", "loop", None),
    ("dLux (loop, jit)",     "#30638e", "--", "D", "dLux",  "loop", True),
    ("dLux (loop, no jit)",  "#8aa9c4", ":",  "v", "dLux",  "loop", False),
    ("dLux (vmap, jit)",     "#30638e", "-",  "X", "dLux",  "vmap", True),
]


def plot_throughput(records):
    """Cases/second vs batch size for the batching strategies (Comp. E)."""
    if not records:
        print("No throughput results in results/throughput/ - run run_throughput.py first.")
        return
    n_star = max((r["n"] for r in records), default=256)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, color, ls, marker, pkg, mode, jit in THRU_CASES:
        rows = sorted((r for r in records if r["package"] == pkg
                       and r["mode"] == mode
                       and (jit is None or r.get("jit") == jit)),
                      key=lambda r: r["batch"])
        if not rows:
            continue
        bs = [r["batch"] for r in rows]
        thru = [r["throughput_per_s"] for r in rows]
        ax.plot(bs, thru, ls=ls, marker=marker, color=color, lw=2, ms=7,
                label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size (independent aberration cases)")
    ax.set_ylabel("throughput [cases / second]  (higher is better)")
    dev = records[0].get("device_used", "cpu")
    ax.set_title(f"Comparison E — batched forward-model throughput\n"
                 f"N = {n_star}, device = {dev}  (loops are flat; vmap scales)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "throughput.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def plot_forward(records):
    """Single forward-model (image-simulation) time vs N, no retrieval (Comp. G)."""
    if not records:
        print("No forward results in results/forward/ - run run_forward.py first.")
        return
    def series(pkg, jit=None):
        rows = sorted((r for r in records if r["package"] == pkg
                       and (jit is None or r.get("jit") == jit)),
                      key=lambda r: r["n"])
        return [r["n"] for r in rows], [r["time_median_s"] * 1e3 for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pkg in ["POPPY", "HCIPy", "prysm", "PROPER"]:
        ns, t = series(pkg)
        if ns:
            c, m = STYLE[pkg]
            ax.plot(ns, t, marker=m, color=c, lw=2, ms=8, label=pkg)
    c = STYLE["dLux"][0]
    ns_j, t_j = series("dLux", jit=True)
    if ns_j:
        ax.plot(ns_j, t_j, marker="D", color=c, lw=2, ms=8, label="dLux (jit)")
    ns_e, t_e = series("dLux", jit=False)
    if ns_e:
        ax.plot(ns_e, t_e, marker="v", color=c, lw=2, ms=7, ls="--",
                label="dLux (no jit)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("pupil pixels across (N)")
    ax.set_ylabel("one forward propagation [ms]  (median)")
    dev = records[0].get("device_used", "cpu")
    ax.set_title("Comparison G — forward-model speed (image simulation only)\n"
                 f"single pupil->PSF, no retrieval; dLux jit vs no-jit, device = {dev}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="propagator", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "comparison_G_forward.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def plot_callcount(records):
    """Bar plot: Python-visible function calls per forward propagation (Comp. F)."""
    if not records:
        print("No call-count results in results/callcount/ - run run_callcount.py first.")
        return
    # One bar per record, sorted fewest-to-most calls.
    recs = sorted(records, key=lambda r: r["calls_per_prop"])
    labels = [r["label"] for r in recs]
    calls = [r["calls_per_prop"] for r in recs]
    colors = [STYLE.get(r["package"], ("#888", ""))[0] for r in recs]
    # eager dLux drawn hollow-ish to set it apart from the jit bar.
    alphas = [0.55 if r.get("label") == "dLux (eager)" else 1.0 for r in recs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, calls, color=colors)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    ax.set_yscale("log")
    ax.set_ylabel("Python-visible function calls per forward propagation")
    n_star = recs[0].get("n", "")
    ax.set_title(f"Comparison F — code layers per image simulation  (N = {n_star})\n"
                 "cProfile call count; lower = leaner path to a PSF")
    for b, c in zip(bars, calls):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()
    out = os.path.join(RESULTS, "callcount.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def plot_summary_bar(records):
    if not records:
        return
    n_star = max(r["n"] for r in records)
    labels, times, colors = [], [], []
    # (package, mode, jit-selector, label)
    order = [("POPPY", "nograd", None, "POPPY\nnograd"),
             ("HCIPy", "nograd", None, "HCIPy\nnograd"),
             ("prysm", "nograd", None, "prysm\nnograd"),
             ("dLux", "nograd", True, "dLux\nnograd jit"),
             ("dLux", "nograd", False, "dLux\nnograd no-jit"),
             ("prysm", "backprop", None, "prysm\nbackprop"),
             ("dLux", "backprop", True, "dLux\nbackprop jit"),
             ("dLux", "backprop", False, "dLux\nbackprop no-jit")]
    for pkg, mode, jit, label in order:
        rows = [r for r in records if r["package"] == pkg and r["mode"] == mode
                and r["n"] == n_star and (jit is None or r.get("jit") == jit)]
        if not rows:
            continue
        labels.append(label)
        times.append(rows[0]["time_median_s"] * 1e3)
        colors.append(STYLE[pkg][0])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, times, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("wall time [ms]  (median)")
    ax.set_title(f"Phase-retrieval runtime at N = {n_star}")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t, f"{t:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "summary_bar.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def write_table(records):
    records = sorted(records, key=lambda r: (r["package"], r["mode"], r["n"]))
    lines = [
        "# Phase-retrieval propagator benchmark — results",
        "",
        f"Machine: {records[0]['platform']} ({records[0]['machine']})" if records else "",
        "",
        "| Package | Mode | JIT | Dtype | N | Device | Median [ms] | Iters | "
        "Fwd evals | Final cost | Phase RMS err [rad] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        jit = {True: "yes", False: "no"}.get(r.get("jit"), "-")
        dtype = r.get("intensity_dtype", "-")
        lines.append(
            f"| {r['package']} | {r['mode']} | {jit} | {dtype} | {r['n']} | "
            f"{r['device_used']} | {r['time_median_s']*1e3:.1f} | {r['n_iter']} | "
            f"{r['n_feval']} | {r['final_cost']:.2e} | "
            f"{r['phase_rms_error_rad']:.4f} |")
    out = os.path.join(RESULTS, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", out)
    print("\n".join(lines))


def main():
    records = load()
    if not records:
        print("No results found in results/. Run run_all.py first.")
        return
    plot_comparison_a(records)
    plot_comparison_b(records)
    plot_jit(records)
    plot_summary_bar(records)
    write_table(records)
    plot_memory(load("mem"))
    plot_throughput(load("throughput"))
    plot_callcount(load("callcount"))
    plot_forward(load("forward"))


if __name__ == "__main__":
    main()
