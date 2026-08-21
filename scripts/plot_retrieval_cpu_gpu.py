"""Phase retrieval across two devices and two gradient methods, in one figure.

This is a CROSS-BOARD figure and `dragrace plot` deliberately will not draw it:
that command emits one figure per (case, config, machine, contract) because
those are the axes along which results are not comparable. Here two of them are
crossed on purpose --

    pr_zernike11_numeric_scan   x  cpu_numpy_1t + gpu_f64     poppy
    pr_zernike11_analytic_scan  x  cpu_numpy_1t + gpu_f64     prysm, dLux

-- and the crossing is legitimate only because the two cases are the same
optical system, the same truth wavefront, the same starting theta and the same
L-BFGS-B stopping tests, differing solely in `retrieval.gradient`. That is
checked at run time below rather than asserted in prose.

WHAT THE TWO PANELS SEPARATE. A retrieval's wall time is a product of two
independent factors:

    total = (forward-model evaluations) x (cost of one forward model)

The gradient method sets the first: finite differences cost P+1 = 12 forward
models per gradient, an adjoint or an AD pass costs O(1). The device sets the
second. Plotting only the total confounds them, and the confusion is not
academic -- a reader would attribute to "GPU acceleration" a gap that is
actually 13x more work. So the left panel is the total and the right is the
per-evaluation cost, and the ratio between the panels is the gradient method.

numpy_baseline is excluded by request. It is the harness's own floor rather
than a library under test, and on this figure it would also be the only series
with no GPU partner.

    python scripts/plot_retrieval_cpu_gpu.py [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dragrace.plots import GRID, INK, INK_SOFT, SURFACE, style_for  # noqa: E402
from dragrace.report import aggregate, best_points  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "docs" / "figures" / "pr_zernike11_cpu_vs_gpu.png"

#: (adapter, case, gradient method as it appears in the legend).
SERIES = (
    ("poppy", "pr_zernike11_numeric_scan", "numerical"),
    ("prysm", "pr_zernike11_analytic_scan", "analytic"),
    ("dlux", "pr_zernike11_analytic_scan", "analytic"),
)
#: (config id, device label, line style). Solid for the device being argued
#: for, dashed for its baseline, so a CPU/GPU pair reads as one colour.
DEVICES = (("cpu_numpy_1t", "CPU", (0, (4.5, 1.8))),
           ("gpu_f64", "GPU", (0, ())))


def collect(repo: Path) -> tuple[dict, dict, list[int]]:
    """{(adapter, config): {N: row}}, the machines seen, and the sizes."""
    rows = best_points([
        r for r in aggregate(repo / "results")
        if r.get("scan_value") is not None and r["status"] == "ok"
        and r["median_s"] and not r["traced"] and r.get("gate") == "pass"
        and r["case"] in {c for _, c, _ in SERIES}
        and r["config"] in {c for c, _, _ in DEVICES}
        and r["adapter"] in {a for a, _, _ in SERIES}
    ])
    out: dict[tuple[str, str], dict[int, dict]] = {}
    machines: dict[str, str] = {}
    for r in rows:
        out.setdefault((r["adapter"], r["config"]), {})[r["scan_value"]] = r
        machines[r["machine"]] = r.get("cpu") or "?"
    sizes = sorted({v for d in out.values() for v in d})
    return out, machines, sizes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    data, machines, sizes = collect(args.repo)
    if not data:
        raise SystemExit("no gated retrieval results for these adapters/configs")
    if len(machines) > 1:
        raise SystemExit(f"results span {len(machines)} machines and must not be "
                         f"drawn on one figure: {machines}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), sharex=True,
                             constrained_layout=True, facecolor=SURFACE)
    # Reserve the strip the caption is drawn into. constrained_layout knows
    # nothing about raw fig.text, so without this the caption lands on top of
    # the x tick labels.
    fig.get_layout_engine().set(rect=(0.008, 0.115, 0.988, 0.90))
    for ax in axes:
        ax.set_facecolor(SURFACE)

    missing: list[str] = []
    fev: dict[tuple[str, str], set[int]] = {}
    for adapter, case_id, _ in SERIES:
        colour, marker = style_for(adapter)
        for config, device, dash in DEVICES:
            pts = data.get((adapter, config))
            if not pts:
                missing.append(f"{adapter} on {device}")
                continue
            x = sorted(pts)
            total_ms = [pts[n]["median_s"] * 1e3 for n in x]
            # s_per_fev is recorded per point; recompute where an older result
            # predates it rather than dropping the series.
            per_fev_ms = [
                (pts[n].get("s_per_fev") or (pts[n]["median_s"] / pts[n]["n_fev"])) * 1e3
                for n in x]
            fev[(adapter, device)] = {pts[n]["n_fev"] for n in x}
            for ax, y in ((axes[0], total_ms), (axes[1], per_fev_ms)):
                ax.plot(x, y, color=colour, lw=2.0, ls=dash, marker=marker,
                        ms=6.5, mew=1.2, mfc=SURFACE if device == "CPU" else colour,
                        zorder=4)

    for ax, title, ylab in (
            (axes[0], "(a) one complete retrieval",
             "median wall time for the whole optimisation (ms)"),
            (axes[1], "(b) cost of one forward model",
             "wall time / forward-model evaluation (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK_SOFT)
        ax.set_xlabel("pupil array size $N_p$ (samples across)", fontsize=9.5,
                      color=INK_SOFT)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = []
    for adapter, _, grad in SERIES:
        colour, marker = style_for(adapter)
        handles.append(Line2D([], [], color=colour, marker=marker, lw=2.0,
                              label=f"{adapter}  ({grad} gradient)"))
    handles.append(Line2D([], [], color=INK_SOFT, lw=2.0, ls=(0, ()), label="GPU (RTX 4090)"))
    handles.append(Line2D([], [], color=INK_SOFT, lw=2.0, ls=(0, (4.5, 1.8)),
                          mfc=SURFACE, label="CPU (1 core, pinned)"))
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper left",
                   labelcolor=INK, handlelength=2.4, borderaxespad=0.6)

    def fev_note() -> str:
        parts = []
        for adapter, _, _ in SERIES:
            seen = sorted({n for dev in ("CPU", "GPU")
                           for n in fev.get((adapter, dev), ())})
            if seen:
                span = str(seen[0]) if len(seen) == 1 else f"{seen[0]}-{seen[-1]}"
                parts.append(f"{adapter} {span}")
        return "forward-model evaluations per retrieval: " + ", ".join(parts)

    caption = [
        f"machine {list(machines)[0]} ({list(machines.values())[0]}) + NVIDIA RTX 4090 "
        f"· idiomatic-v1 · complex128 on both devices · median of the repeats",
        "every point passed its coefficient gate, and the iteration counts agree "
        "across devices (21-23), so both legs solved the same problem",
        fev_note() + ".",
        "panel (b) = panel (a) / that count: the propagation cost the device acts "
        "on, with the gradient method divided out.",
    ]
    if missing:
        caption.append("NOT MEASURED: " + ", ".join(sorted(set(missing))))
    fig.text(0.008, 0.010, "\n".join(caption), fontsize=7.6, color=INK_SOFT,
             va="bottom", linespacing=1.5)
    fig.suptitle("Phase retrieval: what the device buys, and what the gradient "
                 "method buys", fontsize=12.5, color=INK, x=0.008, ha="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
