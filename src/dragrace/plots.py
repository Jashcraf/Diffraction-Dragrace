"""Line plots for scan cases: runtime against array size, one line per adapter.

The scan curve answers a question the bar chart cannot. A single-N ranking is
almost always dispatch-bound at one end and memory-bound at the other, and the
lines cross: at N=128 these codes are separated by Python overhead, at N=2048 by
BLAS. What a reader needs is the *slope* -- an adapter tracking the ideal FLOP
curve is spending its time on the physics, and one steeper than it is doing work
the physics does not require, most often rebuilding kernel matrices per call.
So the ideal-FLOP scaling is drawn on the same axes as a reference, normalised to
the fastest code at the smallest size; it is a slope to compare against, not a
claim about absolute achievable time.

Three rules, each inherited from how this repo reports anything:

  * one figure per machine fingerprint, never a merged axis -- the same rule
    report.py enforces, for the same reason (MKL dispatch differs by CPU vendor).
  * only gated, untimed-by-nobody points are drawn: status ok, gate pass, not
    traced. Excluded points are counted in the caption rather than dropped
    silently, because a curve with a hole in it is a finding.
  * every figure carries its machine, config and repeat count, since a PNG
    outlives the directory it was written in.

Three levels of visual weight, which is what makes a crowded figure readable:

  ideal FLOP scaling   a thick grey slope guide, behind everything.
  numpy_baseline       a soft wide corridor. It is the harness's own floor, not
                       a competitor, so it is a thing to read *against* rather
                       than a seventh line competing for the same pixels.
  the libraries        thin lines with markers, on top.

Colour is assigned per adapter from a fixed order, so an adapter keeps its
colour when another is filtered out, and every series carries a distinct marker
and dash pattern as well. The third encoding is not decoration: these codes
routinely agree to within 2%, so their lines genuinely coincide, and a solid
line drawn over another solid line erases it. Dashes let the one underneath
show through, and a surface-coloured halo keeps the crossing legible.
"""
from __future__ import annotations

from pathlib import Path

from .report import case_kind_for, latest_axes, scan_rows

# Categorical palette, fixed order. Validated for the light surface: worst
# adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
#: Fixed dash pattern per slot, alongside colour and marker. Three encodings for
#: one identity is not belt-and-braces here: these codes routinely agree to
#: within 2%, so their lines coincide, and a solid line drawn over another solid
#: line simply erases it. Dashes let the one underneath show through.
DASHES = ((None, None), (5, 1.6), (1, 1.4), (7, 1.6, 1.2, 1.6),
          (3, 1.2, 3, 1.2, 1, 1.2), (2.5, 1.4), (9, 2), (4, 1.2, 1, 1.2))
OVERFLOW_COLOR = "#6b6a66"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"

#: Slot order for colour assignment. Fixed rather than derived from whatever
#: happens to be in a given result set, so a figure drawn from two adapters and
#: one drawn from six agree on which line is prysm.
ADAPTER_ORDER = ("numpy_baseline", "prysm", "hcipy", "poppy", "lentil", "proper",
                 "dlux", "abcdlux")

#: Drawn as a corridor rather than as a series. It is the harness's own floor --
#: "what does this cost if you just write it down" -- and not a competitor, so
#: giving it the same visual weight as a library both overstates it and costs
#: the libraries the pixels they need to separate from each other.
REFERENCE_ADAPTER = "numpy_baseline"

AXIS_LABELS = {
    "n_pupil": "pupil array size $N_p$ (samples across)",
    "n_focus": "focal grid size $N_f$ (samples across)",
    "n_zernike": "free parameters $P$ (Zernike coefficients solved for)",
}


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:                         # noqa: BLE001
        raise SystemExit(
            f"matplotlib is required for `dragrace plot` ({exc}). It is an optional "
            f'dependency of the harness: pip install -e ".[report]"\n'
            f"The same numbers are available without it: `dragrace report` prints a "
            f"scan table, and results/index.json carries every point."
        ) from exc
    matplotlib.use("Agg")                              # write files, never a window
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    return plt, Patch, pe


def _slot(adapter: str) -> int:
    """The adapter's fixed index into PALETTE / MARKERS / DASHES.

    One function so the three encodings cannot drift apart: a figure that took
    its colour from one rule and its dash from another would give an adapter two
    identities, which is worse than giving it one weak one.
    """
    if adapter in ADAPTER_ORDER:
        return ADAPTER_ORDER.index(adapter)
    # An adapter this module has never heard of still gets a slot, but never by
    # cycling a hue that already means something else.
    return len(ADAPTER_ORDER) + sum(ord(c) for c in adapter) % 4


def style_for(adapter: str) -> tuple[str, str]:
    """(colour, marker) for an adapter, stable across figures."""
    i = _slot(adapter)
    if i < len(PALETTE):
        return PALETTE[i], MARKERS[i]
    return OVERFLOW_COLOR, MARKERS[i % len(MARKERS)]


def dash_for(adapter: str) -> tuple:
    """Matplotlib dash tuple for an adapter, from the same fixed slot.

    The third encoding, and on some boards the load-bearing one: these codes
    routinely agree to within a few percent, so their lines genuinely coincide
    and a solid line drawn over another erases it. It also carries identity for
    a reader who cannot separate two adjacent hues -- the categorical palette
    has one pair (poppy's amber against prysm's orange) whose normal-vision
    separation is below the dE 15 floor, so on any figure showing both, colour
    alone is not enough and is not asked to be.
    """
    on_off = DASHES[_slot(adapter) % len(DASHES)]
    return (0, ()) if on_off[0] is None else (0, on_off)


def scan_series(rows: list[dict], case: str | None = None,
                config: str | None = None) -> dict[tuple, dict[str, list[dict]]]:
    """Plottable rows grouped into figures, then into lines.

    Key is (machine, cpu, contract, case, config, mode, scan_param). Machine
    first because two fingerprints must never share an axis; contract next
    because a curve measured through a library's transform and one measured
    through its documented API are different quantities, however alike the
    numbers look.
    """
    groups: dict[tuple, dict[str, list[dict]]] = {}
    for r in scan_rows(rows, case=case, config=config):
        key = (r["machine"], r["cpu"], r.get("contract"), r["case"], r["config"],
               r["mode"], r["scan_param"])
        groups.setdefault(key, {}).setdefault(r["adapter"], []).append(r)
    for lines in groups.values():
        for pts in lines.values():
            pts.sort(key=lambda r: r["scan_value"])
    return groups


def _excluded(rows: list[dict], case: str, config: str, mode: str, machine: str,
              plotted: set[tuple[str, int]]) -> list[dict]:
    """Genuine holes in this figure: points with no plotted counterpart.

    A point that failed under an older run_id and succeeded under a newer one is
    not a hole -- reporting it would put an adapter in the caption and on the
    chart at the same time, which reads as a contradiction rather than as
    history.
    """
    return [r for r in rows
            if r.get("scan_value") is not None and r["case"] == case
            and r["config"] == config and r["mode"] == mode and r["machine"] == machine
            and (r["status"] != "ok" or not r["median_s"] or r["traced"])
            and (r["adapter"], r["scan_value"]) not in plotted]


#: What one timed iteration actually was, per case kind. The generic label is
#: not merely imprecise on these boards -- it names an operation that did not
#: happen. A phase-retrieval iteration is a whole nonlinear optimisation,
#: hundreds of forward models, which is also why its numbers are seconds where
#: every other board's are milliseconds.
_Y_LABELS = {
    "aperture": "median aperture drawing time (ms)",
    "phase_retrieval": "median time for one complete retrieval (ms)",
}


def _y_label(lines: dict[str, list[dict]], case_id: str) -> str:
    """What the timed region measured, in the axis label.

    `case_kind` is read from the results, and falls back to the case file for
    results written before that field existed rather than mislabelling them.
    """
    flat = [r for pts in lines.values() for r in pts]
    return _Y_LABELS.get(case_kind_for(flat, case_id), "median propagation time (ms)")


#: Reason substrings -> the short phrase put on the chart. A curve that stops
#: early is one of the most easily misread things on a scan plot: without a mark
#: at the end of the line it looks like the code was simply not run, when in
#: fact it was run and could not finish. Matched on the reason rather than the
#: status because "unsupported" covers both "this library cannot do it at all"
#: and "this library cannot do it at THIS size", which are different findings.
_TRUNCATION_PHRASES = (
    ("out of memory", "out of memory"),
    ("oom", "out of memory"),
    ("gib", "out of memory"),
    ("memory", "out of memory"),
    ("timeout", "timed out"),
    ("timed out", "timed out"),
)


def _truncation_phrase(reason: str, status: str) -> str:
    low = (reason or "").lower()
    for needle, phrase in _TRUNCATION_PHRASES:
        if needle in low:
            return phrase
    return {"accuracy_fail": "failed its gate"}.get(status, "no result")


def _annotate_truncated(ax, lines: dict[str, list[dict]], rows: list[dict],
                        case_id: str, config_id: str, mode: str, machine: str,
                        contract: str) -> None:
    """Mark the end of any curve that stops before the widest size on the figure.

    Only curves that stop *early* are annotated, and only where there is a real
    excluded point beyond the last one drawn -- an adapter that was simply never
    run at the larger sizes has no result to explain and gets nothing.
    """
    import matplotlib.patheffects as pe          # matplotlib is an optional dep

    plotted = {(a, r["scan_value"]) for a, pts in lines.items() for r in pts}
    skipped = [r for r in _excluded(rows, case_id, config_id, mode, machine, plotted)
               if r.get("contract") == contract]
    if not skipped:
        return
    widest = max(r["scan_value"] for pts in lines.values() for r in pts)

    for adapter, pts in lines.items():
        last = max(pts, key=lambda r: r["scan_value"])
        if last["scan_value"] >= widest:
            continue
        beyond = [r for r in skipped
                  if r["adapter"] == adapter and r["scan_value"] > last["scan_value"]]
        if not beyond:
            continue
        first_gone = min(beyond, key=lambda r: r["scan_value"])
        phrase = _truncation_phrase(first_gone.get("reason") or "",
                                    first_gone.get("status") or "")
        colour, _ = style_for(adapter)
        ax.annotate(
            f"✕ {phrase} above {last['scan_value']}",
            xy=(last["scan_value"], last["median_s"] * 1e3),
            xytext=(6, -14), textcoords="offset points",
            fontsize=8, color=colour, va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)],
            zorder=6,
        )
        ax.plot([last["scan_value"]], [last["median_s"] * 1e3],
                marker="x", ms=9, mew=2.0, color=colour, zorder=6,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])


def _ideal_reference(lines: dict[str, list[dict]]) -> tuple[list[float], list[float]] | None:
    """Ideal-FLOP scaling through the fastest code's smallest point.

    Anchored to a measurement rather than to a machine peak on purpose: the
    figure is making a claim about *slope*, and anchoring to peak FLOP/s would
    smuggle in a claim about absolute efficiency that belongs to the roofline.
    """
    ideal: dict[int, float] = {}
    for pts in lines.values():
        for r in pts:
            if r["ideal_gflop"]:
                ideal.setdefault(r["scan_value"], r["ideal_gflop"])
    if len(ideal) < 2:
        return None

    sizes = sorted(ideal)
    first = min((r["median_s"] for pts in lines.values() for r in pts
                 if r["scan_value"] == sizes[0]), default=None)
    if not first:
        return None
    return sizes, [first * 1e3 * ideal[n] / ideal[sizes[0]] for n in sizes]


def dashes_for(adapter: str) -> tuple:
    """Dash pattern for an adapter, from the same fixed slot as its colour."""
    if adapter in ADAPTER_ORDER:
        i = ADAPTER_ORDER.index(adapter)
    else:
        i = len(ADAPTER_ORDER) + sum(ord(c) for c in adapter) % 4
    return DASHES[i % len(DASHES)]


def _direct_labels(ax, lines: dict[str, list[dict]], order: list[str],
                   min_separation: float = 0.25) -> None:
    """Label line ends directly, but only when they will not collide.

    Direct labels beat a legend lookup when the curves separate -- which is the
    interesting case, since a scan that separates is a scan with a result. When
    they converge (three MFT codes all sitting on the same BLAS do converge, to
    within a percent) the labels would overlap and say less than the legend
    already does, so they are dropped rather than shuffled into place.
    """
    order = [a for a in order if a != REFERENCE_ADAPTER]
    if not order or len(order) > 4:
        return
    finals = sorted(lines[a][-1]["median_s"] for a in order)
    if any(b / a - 1.0 < min_separation for a, b in zip(finals, finals[1:])):
        return
    for adapter in order:
        last = lines[adapter][-1]
        ax.annotate(adapter, xy=(last["scan_value"], last["median_s"] * 1e3),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, clip_on=False)


def plot_scans(rows: list[dict], out_dir: str | Path, fmt: str = "png",
               case: str | None = None, config: str | None = None,
               linear: bool = False, dpi: int = 160) -> list[Path]:
    """One figure per (machine, case, config, mode). Returns the paths written."""
    plt, Patch, pe = _require_matplotlib()

    groups = scan_series(rows, case=case, config=config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (machine, cpu, contract, case_id, config_id, mode, param), lines in sorted(
            groups.items(), key=lambda kv: str(kv[0])):
        fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=dpi)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        # Two backdrops and then the codes, in that order of visual weight:
        #   ideal FLOP scaling   the physics floor's slope
        #   numpy_baseline       the empirical floor -- "what this costs if you
        #                        just write it down". Not a competitor (see the
        #                        adapter's own docstring), so it is drawn as a
        #                        corridor rather than as a seventh series; the
        #                        codes then read against it instead of fighting
        #                        it for the same few pixels.
        # Suppressed on the aperture board: rasterisation has no arithmetic
        # floor, and the line there was a memory-traffic bound anchored to a
        # measurement -- it carried no information of its own and crossed the
        # curves, inviting the reading that a code was beating a physics floor.
        flat_rows = [r for pts in lines.values() for r in pts]
        ref = (None if case_kind_for(flat_rows, case_id) == "aperture"
               else _ideal_reference(lines))
        if ref is not None:
            ax.plot(ref[0], ref[1], color=GRID, lw=4.0, solid_capstyle="round",
                    zorder=1, label="ideal FLOP scaling")

        # Plotted (and so legended) in descending final time, which is the order
        # the lines appear top to bottom at the right edge.
        order = sorted(lines, key=lambda a: -lines[a][-1]["median_s"])
        for adapter in order:
            pts = lines[adapter]
            colour, marker = style_for(adapter)
            x = [r["scan_value"] for r in pts]
            y = [r["median_s"] * 1e3 for r in pts]
            lo = [(r["min_s"] or r["median_s"]) * 1e3 for r in pts]
            hi = [(r["p95_s"] or r["median_s"]) * 1e3 for r in pts]
            # min-to-p95 band: the spread is part of the measurement, and a
            # median line alone invites reading a 3% difference as real.
            if adapter == REFERENCE_ADAPTER:
                ax.fill_between(x, lo, hi, color=colour, alpha=0.16, lw=0, zorder=1)
                ax.plot(x, y, color=colour, lw=6.0, alpha=0.38, zorder=2,
                        solid_capstyle="round", label=f"{adapter} (reference)")
                continue

            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
            # A surface-coloured halo under each line, so where two curves
            # coincide the upper one reads as a separate stroke instead of
            # simply erasing the one beneath it.
            ax.plot(x, y, color=colour, lw=2.0, marker=marker, ms=6.5,
                    dashes=dashes_for(adapter),
                    mec=SURFACE, mew=1.2, label=adapter, zorder=3,
                    path_effects=[pe.Stroke(linewidth=4.0, foreground=SURFACE),
                                  pe.Normal()])

        _annotate_truncated(ax, lines, rows, case_id, config_id, mode, machine,
                            contract)
        _direct_labels(ax, lines, order)

        sizes = sorted({r["scan_value"] for pts in lines.values() for r in pts})
        if not linear:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()

        ax.set_ylabel(_y_label(lines, case_id), fontsize=10, color=INK_SOFT)
        ax.set_xlabel(AXIS_LABELS.get(param, param), fontsize=10, color=INK_SOFT)
        ax.set_title(f"{case_id} — runtime vs array size", fontsize=12,
                     color=INK, loc="left", pad=12)

        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)

        # A legend is always present: identity is never carried by colour alone,
        # and three of these hues sit below 3:1 against the surface. The spread
        # band gets an entry of its own -- it is a second encoding on the same
        # axes and covers real area, so leaving it to a footnote reads as an
        # unexplained shape.
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=INK_SOFT, alpha=0.18, lw=0))
        labels.append("min–p95 spread")
        ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9,
                  labelcolor=INK, handlelength=1.8, borderaxespad=0.8)

        caption = (f"{config_id} · {mode} · {contract} · machine {machine} ({cpu}) · "
                   f"median of the repeats in each result, band = min to p95")

        # A figure that puts an XLA-backed line next to OpenBLAS-backed ones
        # without saying so invites the conclusion "X is 2x faster than Y on the
        # same backend", which is exactly what it is not.
        # One config per figure, so the config half of the key is constant.
        inert = {a: ax for (a, _), ax in
                 latest_axes([r for pts in lines.values() for r in pts]).items()}
        if inert:
            caption += ("\nconfig axes without effect: "
                        + "; ".join(f"{a} ({', '.join(sorted(ax))})"
                                    for a, ax in sorted(inert.items())))
            # The strong warning belongs only where the engines actually differ:
            # some lines on a backend the config named and others on their own.
            mixed = {a for a, ax in inert.items() if "fft" in ax}
            if mixed and mixed != set(lines):
                caption += (f" — {', '.join(sorted(mixed))} "
                            f"{'use' if len(mixed) > 1 else 'uses'} a different engine "
                            f"from the rest; not a like-for-like backend comparison")
        # A retrieval board can legitimately mix algorithm classes -- PROPER has
        # no matrix-DFT path at all -- and an FFT-based code cannot choose its
        # focal sampling, so it computes the entire plane and crops. Without
        # this line a reader takes its curve for a slower propagator, when a
        # large part of what separates it is that it was asked for 4096 samples
        # and had to compute four million.
        over = {}
        for adapter, pts in lines.items():
            ratios = [r["focal_computed"] / r["focal_requested"] for r in pts
                      if r.get("focal_computed") and r.get("focal_requested")
                      and r["focal_computed"] > r["focal_requested"]]
            if ratios:
                over[adapter] = max(ratios)
        if over:
            caption += ("\n" + "; ".join(
                f"{a} computes {v:.0f}x the focal samples this case asks for "
                f"(FFT sampling is set by beam/grid, so the whole plane is "
                f"computed and cropped)" for a, v in sorted(over.items())))

        plotted = {(a, r["scan_value"]) for a, pts in lines.items() for r in pts}
        skipped = [r for r in _excluded(rows, case_id, config_id, mode, machine, plotted)
                   if r.get("contract") == contract]
        if skipped:
            # Named, by adapter and by size: a curve that is missing a code
            # entirely is a different finding from one missing its largest point.
            by_adapter: dict[tuple[str, str], list[int]] = {}
            for r in skipped:
                by_adapter.setdefault((r["adapter"], r["status"]), []).append(r["scan_value"])
            what = "; ".join(
                f"{a} {st} at {','.join(str(v) for v in sorted(vs))}"
                for (a, st), vs in sorted(by_adapter.items()))
            caption += f"\nnot plotted: {what}"
        fig.text(0.012, 0.015, caption, fontsize=7.5, color=INK_SOFT, va="bottom")
        # Reserve room from the caption's actual line count -- a fixed margin
        # left the provenance line sitting on top of the x-axis label as soon as
        # a second line appeared.
        # Sized from the caption's line count -- a fixed margin either sat the
        # provenance line on the axis label or left a band of empty page.
        fig.tight_layout(rect=(0, 0.035 + 0.022 * (caption.count("\n") + 1), 1, 1))

        stem = (f"scan_{case_id}_{config_id}_{mode}_{contract}_"
                f"{machine.replace(':', '_')}")
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    return written
