"""Retrieval runtime against the NUMBER OF FREE PARAMETERS, against time budgets.

    python scripts/plot_retrieval_zernike.py [--out PATH] [--degree 1]

The board is cases/phase_retrieval/pr_nzernike_n256_{numeric,analytic}_scan: a
256x256 pupil held fixed while P, the number of Zernike coefficients the
retrieval solves for, grows logarithmically from 3 to 231. Two cases rather than
one because an adapter appears on the board matching the gradient it can supply,
and the pair is identical in every other value -- so the two curves on this
figure are the same optical system, the same truth wavefront and the same
stopping tests, differing only in where the gradient comes from.

WHY THIS AXIS IS THE INTERESTING ONE. Jurling & Fienup (2014), JOSA A 31(7)
1348, argue that the gradient is the whole cost story: a finite-difference
gradient costs P+1 forward models and an analytic one costs O(1). On a
fixed-P board that ratio is a constant and shows up as a vertical offset. Here it
is the SLOPE, so the boards separate as the figure goes right, and the gap
between the fitted exponents is the argument stated as a number.

THE HORIZONTAL LINES are budgets a person actually feels -- a second, a coffee,
an afternoon, a day -- and they are what turns an abstract exponent into "this
code stops being usable at about here".

THE DASHED CONTINUATIONS ARE A FIT, NOT A MEASUREMENT, and the figure says so in
as many places as it can. Nothing on this board was run for 24 hours; the
harness gives each scan point a bounded budget (execution.timeout_s), records a
point that overruns as `timeout`, and this script fits the measured points and
extrapolates through the hole. A power law -- degree 1 in log-log -- is the
default because it is the model the physics predicts (cost ~ P^2 numerically,
P^1 analytically) and because its single parameter is quotable. --degree 2
allows curvature and should be treated with suspicion: a quadratic in log-log
bends without bound, and extrapolating one two decades past its data is a way
of generating a confident wrong number.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dragrace.plots import (GRID, INK, INK_SOFT, SURFACE, dash_for,  # noqa: E402
                            style_for)
from dragrace.report import aggregate, best_points  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "docs" / "figures" / "pr_nzernike_runtime_vs_parameters.png"

NUMERIC = "pr_nzernike_n256_numeric_scan"
ANALYTIC = "pr_nzernike_n256_analytic_scan"
CONFIG = "gpu_f64"

#: (adapter, case, how the gradient is obtained -- as it appears in the legend).
SERIES = (
    ("poppy", NUMERIC, "numerical, finite differences"),
    ("prysm", ANALYTIC, "analytic, prysm adjoint API"),
    ("dlux", ANALYTIC, "analytic, jax.value_and_grad"),
)

#: The budgets, and where each label sits. Chosen to be felt rather than to be
#: round in seconds.
BUDGETS = ((1.0, "1 second"), (300.0, "5 minutes"),
           (2700.0, "45 minutes"), (86400.0, "24 hours"))
DAY_S = 86400.0


def collect(repo: Path) -> tuple[dict, dict, dict]:
    """({(adapter): {P: row}}, {machine: cpu}, {(adapter): [unmeasured rows]})."""
    wanted_cases = {c for _, c, _ in SERIES}
    wanted_adapters = {a for a, _, _ in SERIES}
    rows = best_points([
        r for r in aggregate(repo / "results")
        if r.get("scan_param") == "n_zernike"
        and r["case"] in wanted_cases and r["config"] == CONFIG
        and r["adapter"] in wanted_adapters
    ])

    measured: dict[str, dict[int, dict]] = {}
    machines: dict[str, str] = {}
    for r in rows:
        # The same three exclusions the timing table applies. A retrieval that
        # failed its coefficient gate did not solve the problem the other rows
        # solved, so its wall time is not on the same axis as theirs.
        if (r["status"] == "ok" and r["median_s"] and not r["traced"]
                and r.get("gate") == "pass"):
            measured.setdefault(r["adapter"], {})[r["scan_value"]] = r
            machines[r["machine"]] = r.get("cpu") or "?"

    # A hole in the curve is explained by the LATEST attempt at that point, not
    # the first one ever recorded -- which is what best_points keeps, and is
    # wrong here. A point re-run after a harness fix has both an old failure and
    # a new one on disk, and the caption must say why it is missing TODAY: this
    # board's P=231 row read "CUDA_ERROR_INVALID_IMAGE" from a run that predated
    # the runner's CONDA_PREFIX fix, when the current answer is a clean timeout.
    unmeasured: dict[str, list[dict]] = {}
    latest: dict[tuple[str, int], dict] = {}
    for r in aggregate(repo / "results"):
        if (r.get("scan_param") != "n_zernike" or r["case"] not in wanted_cases
                or r["config"] != CONFIG or r["adapter"] not in wanted_adapters):
            continue
        key = (r["adapter"], r["scan_value"])
        if r["scan_value"] in measured.get(r["adapter"], {}):
            continue                       # this point did measure; not a hole
        seen = latest.get(key)
        if seen is None or (r.get("utc") or "") >= (seen.get("utc") or ""):
            latest[key] = r
    for (adapter, _), r in sorted(latest.items(), key=lambda kv: kv[0]):
        unmeasured.setdefault(adapter, []).append(r)
    return measured, machines, unmeasured


def fit_power_law(p: np.ndarray, t: np.ndarray, degree: int) -> np.poly1d:
    """Least squares in log10-log10. Degree 1 is a power law t = A P^k."""
    return np.poly1d(np.polyfit(np.log10(p), np.log10(t), degree))


def crossing(poly: np.poly1d, p_lo: float, p_hi: float, target: float) -> float | None:
    """Smallest P in (p_lo, p_hi] where the fit reaches `target` seconds.

    Solved on a dense log grid rather than in closed form so that --degree 2
    works too, and so a non-monotone fit reports its FIRST crossing rather than
    whichever root the algebra happens to return.
    """
    grid = np.logspace(np.log10(p_lo), np.log10(p_hi), 4000)
    y = 10.0 ** poly(np.log10(grid))
    above = np.nonzero(y >= target)[0]
    return float(grid[above[0]]) if above.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--degree", type=int, default=1,
                    help="polynomial degree in log-log space (1 = power law)")
    ap.add_argument("--extrapolate-factor", type=float, default=100.0,
                    help="how far past the largest measured P the fit is drawn")
    args = ap.parse_args()

    data, machines, unmeasured = collect(args.repo)
    if not data:
        raise SystemExit(
            "no gated n_zernike retrieval points found. Run the board first:\n"
            "  dragrace sweep --cases pr_nzernike_n256_analytic_scan "
            "pr_nzernike_n256_numeric_scan --configs gpu_f64 "
            "--adapters prysm dlux poppy")
    if len(machines) > 1:
        raise SystemExit(f"results span {len(machines)} machines and must not share "
                         f"one axis: {machines}")

    all_p = sorted({p for d in data.values() for p in d})
    p_max_drawn = max(all_p) * args.extrapolate_factor

    fig, ax = plt.subplots(figsize=(11.6, 7.2), constrained_layout=True,
                           facecolor=SURFACE)
    # Reserve the strip the caption is drawn into; constrained_layout knows
    # nothing about a raw fig.text and would let it land on the tick labels.
    # NOTE the argument order: matplotlib's layout rect is (left, bottom, WIDTH,
    # HEIGHT), not (left, bottom, right, top). Reading it the second way puts
    # the top edge above 1.0 and silently clips the top of the axes off the
    # canvas -- which on this figure removed the 24-hour line and the crossing
    # marker, i.e. the two marks the figure exists to show.
    fig.get_layout_engine().set(rect=(0.008, 0.280, 0.984, 0.660))
    ax.set_facecolor(SURFACE)

    # ---- budgets, behind everything -------------------------------------
    for seconds, label in BUDGETS:
        ax.axhline(seconds, color=INK_SOFT, lw=0.9, ls=(0, (6, 3)), zorder=1,
                   alpha=0.55)
        ax.text(0.998, seconds, f" {label} ", fontsize=8.6, color=INK_SOFT,
                va="center", ha="right", zorder=3,
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6))

    notes: list[str] = []
    fev_note: list[str] = []
    handles: list[Line2D] = []

    label_at: list[tuple[str, float, float]] = []
    for adapter, _case, how in SERIES:
        pts = data.get(adapter)
        colour, marker = style_for(adapter)
        if not pts:
            notes.append(f"{adapter}: no gated points")
            continue
        p = np.array(sorted(pts), dtype=float)
        t = np.array([pts[int(v)]["median_s"] for v in p], dtype=float)

        ax.plot(p, t, color=colour, lw=2.2, ls=dash_for(adapter), marker=marker,
                ms=8.0, mew=1.5, mec=colour, mfc=SURFACE, zorder=5)

        label = f"{adapter}  ({how})"
        if p.size >= args.degree + 1:
            poly = fit_power_law(p, t, args.degree)
            grid = np.logspace(np.log10(p[-1]), np.log10(p_max_drawn), 400)
            ax.plot(grid, 10.0 ** poly(np.log10(grid)), color=colour, lw=1.25,
                    ls=(0, (1.6, 2.6)), alpha=0.8, zorder=4)
            exponent = poly.coefficients[-2] if args.degree == 1 else None
            p_day = crossing(poly, p[-1], p_max_drawn, DAY_S)
            if exponent is not None:
                label += f"   $t \\propto P^{{{exponent:.2f}}}$"
            if p_day is not None:
                ax.plot([p_day], [DAY_S], marker="o", ms=9, mfc=SURFACE,
                        mec=colour, mew=2.0, zorder=6)
                ax.annotate(f"24 h at $P \\approx {p_day:,.0f}$",
                            xy=(p_day, DAY_S), xytext=(6, -16),
                            textcoords="offset points", fontsize=8.6,
                            color=INK, zorder=6)
                notes.append(f"{adapter} reaches 24 h at P ~ {p_day:,.0f} "
                             f"(fit, {p_day / p[-1]:.0f}x past its last measured point)")
            else:
                notes.append(f"{adapter} does not reach 24 h below "
                             f"P = {p_max_drawn:,.0f} on the fit")

        # Direct labels are placed after the loop, once every curve's endpoint
        # is known: identity must not be colour-alone, and which label goes
        # above and which below has to follow the CURVES rather than the draw
        # order, or the two analytic codes -- which agree to within 25% -- get
        # labels that cross over each other.
        label_at.append((adapter, float(p[-1]), float(t[-1])))
        handles.append(Line2D([], [], color=colour, marker=marker, lw=2.2,
                              ls=dash_for(adapter), mfc=SURFACE, mew=1.5,
                              label=label))

        fevs = [pts[int(v)].get("n_fev") for v in p]
        if all(f for f in fevs):
            fev_note.append(f"{adapter} {min(fevs):,}-{max(fevs):,}")

        for r in unmeasured.get(adapter, []):
            why = " ".join((r.get("reason") or r["status"]).split())
            if len(why) > 150:                 # back off to a word boundary
                why = why[:150].rsplit(" ", 1)[0] + " ..."
            notes.append(f"{adapter} P={r['scan_value']} NOT MEASURED "
                         f"({r['status']}): {why}")

    # Highest curve first; a label whose neighbour is within a factor of two
    # goes below its point instead of above.
    label_at.sort(key=lambda r: -r[2])
    previous_t: float | None = None
    for adapter, p_last, t_last in label_at:
        below = previous_t is not None and previous_t / t_last < 2.0
        ax.annotate(adapter, xy=(p_last, t_last), xytext=(10, -16 if below else 11),
                    textcoords="offset points", fontsize=9.8, color=INK,
                    fontweight="bold", zorder=7,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
        previous_t = t_last

    ax.set_xscale("log")
    ax.set_yscale("log")
    # The measured values, plus round decades through the extrapolated region so
    # a reader can place the 24 h crossing without counting gridlines.
    decades = [d for d in (1_000, 10_000, 100_000) if d <= p_max_drawn]
    ticks = sorted(set(all_p) | set(decades))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:,}" if v >= 1000 else str(v) for v in ticks])
    ax.set_xlim(all_p[0] * 0.72, p_max_drawn * 1.3)
    # Room above the 24 h line for its label, and a little below the fastest
    # point, so neither the budget rules nor the data sit on an axis edge.
    t_min = min(r["median_s"] for d in data.values() for r in d.values())
    ax.set_ylim(t_min / 4.0, DAY_S * 6.0)
    ax.set_xlabel("free parameters $P$  (Zernike coefficients solved for)",
                  fontsize=10, color=INK_SOFT)
    ax.set_ylabel("median wall time for one complete retrieval (s)",
                  fontsize=10, color=INK_SOFT)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(handles=handles, frameon=False, fontsize=9.4, loc="lower right",
              labelcolor=INK, handlelength=3.2, borderaxespad=1.0)

    machine, cpu = next(iter(machines.items()))
    caption = [
        f"machine {machine} ({cpu}) + NVIDIA RTX 4090 · config {CONFIG} · "
        f"complex128 · 256x256 pupil at every point · median of the repeats",
        "MARKED, heavy = measured. Thin unmarked continuation = least-squares "
        "fit in log-log, extrapolated past the data; nothing here was run for "
        "24 hours and no point was allowed past its execution.timeout_s budget. "
        "Each code's own dash pattern is its identity, not its status.",
    ]
    if fev_note:
        caption.append("forward-model evaluations per retrieval: "
                       + ", ".join(fev_note)
                       + " — the numerical board pays P+1 of them per gradient "
                         "and the analytic boards O(1), which is the whole "
                         "difference in slope.")
    caption += notes
    wrapped: list[str] = []
    for line in caption:
        wrapped += textwrap.wrap(line, width=176, subsequent_indent="    ") or [""]
    fig.text(0.008, 0.010, "\n".join(wrapped), fontsize=7.6, color=INK_SOFT,
             va="bottom", linespacing=1.55)
    fig.suptitle("Phase retrieval: what one more free parameter costs",
                 fontsize=13, color=INK, x=0.008, y=0.988, ha="left", va="top")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for n in notes:
        print("  " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
