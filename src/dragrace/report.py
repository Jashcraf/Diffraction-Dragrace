"""Aggregation and reporting.

One rule enforced here rather than left to discipline: results from different
machine fingerprints are never combined onto one axis. On this suite that is
not pedantry -- MKL's dispatch differs by CPU vendor, so the same board can
legitimately invert between an AMD and an Intel host, and a merged plot would
present that as a property of the propagators.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _row(r: dict, blocks: dict, path: Path, scan_param: str | None = None) -> dict:
    """One table row.

    `r` carries what the whole file shares (adapter, machine, config); `blocks`
    carries one measurement's own timing/accuracy/flops. For a plain result the
    two are the same dict, for a scan point `blocks` is the point -- which is
    what keeps a five-point scan from collapsing into one row whose median
    belongs to no particular array size.
    """
    t = blocks.get("timing", {}) or {}
    stats = t.get("device_compute_stats", {}) or {}
    acc = blocks.get("accuracy", {}) or {}
    fl = blocks.get("flops", {}) or {}
    return {
        "adapter": (r.get("adapter") or {}).get("name"),
        "case": r.get("case_id"),
        "config": r.get("config_id"),
        "mode": r.get("mode"),
        "status": blocks.get("status", r.get("status")),
        # An aperture board measures drawing, not propagation, and the figures
        # need to know which. Absent from results written before that board
        # existed, so readers must tolerate None.
        "case_kind": r.get("case_kind"),
        "machine": (r.get("machine") or {}).get("id"),
        "cpu": (r.get("machine") or {}).get("cpu"),
        # Results measured under different contracts are not comparable; the
        # marker is absent on anything written before contracts existed.
        "contract": r.get("measurement_contract", "primitive-v1"),
        "grid_centering": (r.get("adapter") or {}).get("grid_centering"),
        # Config axes the adapter declared it cannot honour. A row with any of
        # these is a valid measurement of the code, but not a data point along
        # that axis.
        "axes_not_selectable": (r.get("backend") or {}).get("axes_not_selectable") or [],
        "utc": (r.get("provenance") or {}).get("utc"),
        "scan_param": scan_param,
        "scan_value": blocks.get("scan_value"),
        "median_s": stats.get("median"),
        "min_s": stats.get("min"),
        "p95_s": stats.get("p95"),
        "iqr_s": stats.get("iqr"),
        "traced": t.get("traced", False),
        # Cores the timed region actually used. `threads` in the config is a
        # request; this is what happened. Absent on results written before the
        # measurement existed, so readers must tolerate None.
        "cpu_wall_ratio": t.get("cpu_wall_ratio"),
        "threads_requested": (r.get("backend") or {}).get("requested", {}).get("threads"),
        "rel_l2": acc.get("rel_l2"),
        "gate": acc.get("gate"),
        "ideal_gflop": (fl.get("ideal") or {}).get("flops", 0) / 1e9 or None,
        "ledger_gflop": (fl.get("ledger") or {}).get("flops_total", 0) / 1e9 or None,
        "A_overhead": fl.get("algorithmic_overhead"),
        "build_s": (blocks.get("setup") or {}).get("build_s"),
        "first_call_s": (blocks.get("setup") or {}).get("first_call_s"),
        "mem_peak_mib": ((blocks.get("memory") or {}).get("tracemalloc_peak_bytes") or 0) / 2**20
                        or None,
        "grad_gate": (blocks.get("gradient_accuracy") or {}).get("gate"),
        # Phase-retrieval board. Wall time alone is unreadable there: a row is
        # (forward-model evaluations x cost per evaluation) and the two factors
        # are independent, so both travel with the timing.
        "n_iterations": (blocks.get("retrieval") or {}).get("n_iterations"),
        "n_fev": (blocks.get("retrieval") or {}).get("n_fev"),
        "s_per_fev": (blocks.get("retrieval") or {}).get("seconds_per_forward_model"),
        "loss_reduction": (blocks.get("retrieval") or {}).get("loss_reduction"),
        # Focal samples this code evaluated, against the number the case asked
        # for. An MFT evaluates exactly the requested grid; an FFT-based code
        # cannot, because its focal sampling is set by beam/grid, so it computes
        # the whole plane and crops. Carried so a mixed board says so.
        "focal_computed": (blocks.get("retrieval") or {}).get("focal_points_computed"),
        "focal_requested": (blocks.get("retrieval") or {}).get("focal_points_requested"),
        "retrieval_converged": (blocks.get("retrieval") or {}).get("converged"),
        "forward_rel_l2": (blocks.get("forward_accuracy") or {}).get("rel_l2"),
        "opd_sign": (blocks.get("forward_accuracy") or {}).get("opd_sign_convention"),
        "reason": blocks.get("reason", r.get("reason")),
        "path": str(path),
    }


def aggregate(results_root: str | Path) -> list[dict]:
    rows = []
    for p in sorted(Path(results_root).rglob("result.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:                              # noqa: BLE001
            continue
        scan = r.get("scan") or {}
        if scan.get("points"):
            rows += [_row(r, pt, p, scan.get("parameter")) for pt in scan["points"]]
        else:
            rows.append(_row(r, r, p))
    return rows


def best_points(rows: list[dict]) -> list[dict]:
    """One row per (machine, case, config, mode, adapter, size).

    A scan point re-measured across several run_ids is kept at its *fastest*
    median rather than its latest: noise adds time to a measurement, it does not
    remove it, so the minimum is the least contaminated estimate of what the
    code costs. A curve must show one point per size -- two would draw a line
    that zigzags between run days and invite reading scheduler noise as a
    property of the propagator. Points that never measured (a failed gate, an
    unsupported class) keep their first occurrence so the hole stays visible.
    """
    best: dict[tuple, dict] = {}
    for r in rows:
        if r.get("scan_value") is None:
            continue
        key = (r["machine"], r.get("contract"), r["case"], r["config"], r["mode"],
               r["adapter"], r["scan_value"])
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        new_ok = r["status"] == "ok" and r["median_s"]
        cur_ok = cur["status"] == "ok" and cur["median_s"]
        if new_ok and (not cur_ok or r["median_s"] < cur["median_s"]):
            best[key] = r
    return list(best.values())


def scan_rows(rows: list[dict], case: str | None = None,
              config: str | None = None) -> list[dict]:
    """Plottable scan points: measured, gated, untraced, one per size.

    Applies the same three exclusions as the timing table, because a scan curve
    is a timing comparison drawn sideways and must not relax them.
    """
    sel = [r for r in rows
           if r.get("scan_value") is not None
           and r["status"] == "ok" and r["median_s"] and not r["traced"]
           and (case is None or r["case"] == case)
           and (config is None or r["config"] == config)]
    return best_points(sel)


def latest_axes(rows: list[dict]) -> dict[tuple, set]:
    """Declared inapplicable axes per (adapter, config), newest run wins.

    Keyed by config as well as adapter because the answer depends on both:
    dLux cannot honour a `fft=numpy` config, but on `cpu_xla_1t` the config and
    the adapter agree and only blas/threads remain inapplicable.

    Not the union across runs: the declaration is metadata about the adapter,
    not a measurement, so an older run's broader claim is simply out of date.
    Unioning would leave a superseded caveat on a figure forever -- which is how
    a plot ends up warning about a mixed-backend comparison that is not.
    """
    newest: dict[tuple, tuple] = {}
    for r in rows:
        key = (r["adapter"], r["config"])
        stamp = (r.get("utc") or "", r.get("path") or "")
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, set(r.get("axes_not_selectable") or []))
    return {k: ax for k, (_, ax) in newest.items() if ax}


def case_kind_for(rows: list[dict], case_id: str) -> str | None:
    """The `kind` of a case, from the results or failing that from the case file.

    Results written before the aperture board existed carry no `case_kind`, and
    they also carry an `ideal.flops` for it that has since been withdrawn -- N^2
    "one write per pixel", which was a memory-traffic bound presented as an
    arithmetic floor. Rather than reprice those files or re-run hours of
    measurement, the readers below suppress the ideal line and the ideal row for
    aperture cases outright. The kind is the honest discriminator, so it is
    recovered from the case file when a result predates the field.
    """
    for r in rows:
        if r.get("case") == case_id and r.get("case_kind"):
            return r["case_kind"]
    try:
        import yaml
        for p in Path("cases").rglob(f"{case_id}.yaml"):
            return (yaml.safe_load(p.read_text()) or {}).get("kind")
    except Exception:                                  # noqa: BLE001
        pass
    return None


def _case_label(r: dict) -> str:
    return r["case"] if r.get("scan_value") is None else f"{r['case']}@{r['scan_value']}"


def render_scans(rows: list[dict]) -> list[str]:
    """Scan cases as adapter x size, which is the plot in text form.

    Worth rendering even though `dragrace plot` exists: matplotlib is an
    optional dependency, and the ratio between adjacent columns is the number
    the curve is actually being read for -- 4x per doubling is the quadratic
    MFT floor, anything steeper is work the physics does not require.
    """
    pts = best_points(rows)
    if not pts:
        return []

    out: list[str] = []
    keys = {(r["case"], r["config"], r["scan_param"], r.get("contract")) for r in pts}
    for (case, config, param, contract) in sorted(keys, key=str):
        sel = [r for r in pts if r["case"] == case and r["config"] == config
               and r.get("contract") == contract]
        sizes = sorted({r["scan_value"] for r in sel})
        adapters = sorted({r["adapter"] for r in sel})
        by = {(r["adapter"], r["scan_value"]): r for r in sel}

        out.append(f"\nSCAN {case} [{config}] [{contract}]  {param}, median ms")
        out.append(f"{'adapter':<16}" + "".join(f"{n:>11}" for n in sizes))
        out.append("-" * (16 + 11 * len(sizes)))
        for a in adapters:
            cells = []
            for n in sizes:
                r = by.get((a, n))
                if r is None:
                    cells.append(f"{'-':>11}")
                elif r["status"] != "ok" or not r["median_s"]:
                    cells.append(f"{r['status'][:10]:>11}")
                else:
                    cells.append(f"{r['median_s'] * 1e3:>11.3f}")
            out.append(f"{a:<16}" + "".join(cells))

        ideal = [next((r["ideal_gflop"] for r in sel
                       if r["scan_value"] == n and r["ideal_gflop"]), None) for n in sizes]
        # Never for an aperture board: drawing a pupil has no arithmetic floor,
        # so any figure here is invented. See case_kind_for.
        if case_kind_for(pts, case) != "aperture" and all(i for i in ideal):
            out.append(f"{'ideal GFLOP':<16}" + "".join(f"{i:>11.3f}" for i in ideal))
    return out


def render_retrievals(rows: list[dict]) -> list[str]:
    """The phase-retrieval board, decomposed into its two independent factors.

    Wall time on this board is (forward-model evaluations) x (cost per
    evaluation), and a reader cannot tell those apart from the total. A code can
    lose on time while winning per evaluation -- it needed more of them -- which
    is a completely different finding from being slow, and on the numerical
    board it is the finding: a finite-difference gradient costs P+1 evaluations
    where an analytic one costs O(1).

    `fwd` is the untimed forward-model check: this code's own PSF at the truth
    coefficients against the harness reference. It is what separates "converged
    fast" from "converged fast onto its own private physics". `s` is the OPD
    sign convention the code was found to use, which does not affect the
    recovered coefficients but does affect the sign of any wavefront read out of
    it.
    """
    pts = [r for r in best_points(rows) if r.get("n_fev")]
    if not pts:
        return []

    out: list[str] = []
    keys = {(r["case"], r["config"], r.get("contract")) for r in pts}
    for case, config, contract in sorted(keys, key=str):
        sel = [r for r in pts if r["case"] == case and r["config"] == config
               and r.get("contract") == contract]
        out.append(f"\nRETRIEVAL {case} [{config}] [{contract}]")
        out.append(f"{'adapter':<16}{'N':>6}{'total ms':>11}{'iters':>7}{'n_fev':>7}"
                   f"{'ms/fev':>9}{'cores':>7}{'loss drop':>11}{'fwd':>10}{'s':>3}"
                   f"{'focal x':>9}  gate")
        out.append("-" * 103)
        for r in sorted(sel, key=lambda r: (r["adapter"], r["scan_value"] or 0)):
            per = f"{r['s_per_fev'] * 1e3:.3f}" if r.get("s_per_fev") else "-"
            drop = f"{r['loss_reduction']:.1e}" if r.get("loss_reduction") else "-"
            fwd = f"{r['forward_rel_l2']:.1e}" if r.get("forward_rel_l2") is not None else "-"
            cores = (f"{r['cpu_wall_ratio']:.2f}"
                     if r.get("cpu_wall_ratio") is not None else "-")
            over = "-"
            if r.get("focal_computed") and r.get("focal_requested"):
                over = f"{r['focal_computed'] / r['focal_requested']:.0f}x"
            out.append(f"{r['adapter']:<16}{r['scan_value'] or 0:>6}"
                       f"{(r['median_s'] or 0) * 1e3:>11.1f}"
                       f"{r.get('n_iterations') or 0:>7}{r['n_fev']:>7}{per:>9}"
                       f"{cores:>7}{drop:>11}{fwd:>10}"
                       f"{(r.get('opd_sign') or '?'):>3}{over:>9}"
                       f"  {r['gate'] or '-'}")

        over = {r["adapter"]: r["focal_computed"] / r["focal_requested"]
                for r in sel if r.get("focal_computed") and r.get("focal_requested")
                and r["focal_computed"] > r["focal_requested"]}
        if over:
            worst = max(over.values())
            out.append(
                f"  focal x: focal samples computed / asked for. "
                f"{', '.join(sorted(over))} "
                f"{'are' if len(over) > 1 else 'is'} FFT-based, so the focal sampling "
                f"is set by beam/grid and the\n     whole plane must be computed and "
                f"cropped -- up to {worst:.0f}x the samples the case pins, "
                f"{100 * (1 - 1 / worst):.2f}% of it discarded. That is a "
                f"CAPABILITY\n     difference from the matrix-DFT codes, which "
                f"evaluate only the grid asked for, and it is not the same claim as "
                f"'this propagator is slower'.")
        signs = {r["adapter"]: r.get("opd_sign") for r in sel if r.get("opd_sign")}
        flipped = sorted(a for a, s in signs.items() if s == "-")
        if flipped:
            out.append(f"  s: OPD sign convention. {', '.join(flipped)} "
                       f"{'carry' if len(flipped) > 1 else 'carries'} "
                       f"exp(-2i.pi.OPD/lambda), opposite to the harness and to the "
                       f"other codes here.\n     Harmless for the retrieval -- each "
                       f"code fits a PSF its own model produced, so both conventions "
                       f"recover the same\n     theta -- but a wavefront read out of "
                       f"{'these' if len(flipped) > 1 else 'this'} "
                       f"code{'s' if len(flipped) > 1 else ''} has the opposite sign.")
    return out


def render_text(rows: list[dict]) -> str:
    # Grouped by contract as well as machine, and for the same reason: a
    # primitive-v1 row measured the library's transform, an idiomatic-v1 row
    # measured the call its documentation teaches, and putting them in one table
    # would present an API-surface difference as a propagation difference.
    by_machine = defaultdict(list)
    for r in rows:
        by_machine[(r["machine"], r["cpu"], r.get("contract"))].append(r)

    out = []
    contracts = {k[2] for k in by_machine}
    for (mid, cpu, contract), rs in sorted(by_machine.items(), key=lambda kv: str(kv[0])):
        out.append(f"\n=== machine {mid}  {cpu}   [{contract}] ===")
        timed = [r for r in rs if r["status"] == "ok" and r["median_s"] and not r["traced"]]
        if timed:
            out.append(f"\n{'adapter':<16}{'case':<24}{'config':<16}"
                       f"{'median ms':>11}{'GFLOP':>9}{'GFLOP/s':>10}{'A':>7}  gate")
            out.append("-" * 100)
            # Scan points sort by size first, so a scan reads down the table as
            # a curve rather than as adapters interleaved by wall time.
            for r in sorted(timed, key=lambda r: (r["case"], r["config"],
                                                  r["scan_value"] or 0, r["median_s"])):
                g = r["ideal_gflop"] or 0
                rate = g / r["median_s"] if r["median_s"] else 0
                a = f"{r['A_overhead']:.2f}" if r["A_overhead"] else "-"
                out.append(f"{r['adapter']:<16}{_case_label(r):<24}{r['config']:<16}"
                           f"{r['median_s'] * 1e3:>11.3f}{g:>9.3f}{rate:>10.2f}{a:>7}  "
                           f"{r['gate'] or '-'}")

        out += render_scans(rs)

        # Named per block rather than per row: it is a property of the adapter
        # on this config, and a reader needs it before reading any of the times.
        # Union per adapter: an adapter declaring more axes in a later run must
        # not appear twice, which would read as two different adapters.
        # A row that used more cores than it asked for is not a slow-or-fast
        # question, it is a labelling failure: every other row on the same axis
        # was held to the request. Surfaced per block, above the axis notes,
        # because it invalidates comparisons rather than merely qualifying them.
        overs: dict[str, float] = {}
        for r in rs:
            got, want = r.get("cpu_wall_ratio"), r.get("threads_requested")
            if got and want and got > 1.5 * want:
                overs[r["adapter"]] = max(overs.get(r["adapter"], 0.0), got / want)
        if overs:
            out.append("\nTHREAD REQUEST NOT HONOURED")
            for adapter, factor in sorted(overs.items()):
                out.append(f"  {adapter:<16}used {factor:.1f}x the requested cores")
            out.append("  These rows are not comparable with the rest of their board: the "
                       "others were held\n  to the request. Measured from process CPU time "
                       "over the timed region, not assumed.")

        inert = latest_axes(rs)
        if inert:
            out.append("\nBACKEND AXES THAT DID NOT APPLY")
            for (adapter, cfg), axes in sorted(inert.items()):
                out.append(f"  {adapter:<16}{cfg:<16}{', '.join(sorted(axes))}")
            out.append("  These rows measure the code honestly but are not data points "
                       "along those\n  axes -- the library has no such knob, or the "
                       "harness cannot verify it.")

        out += render_retrievals(rs)

        grad = [r for r in rs if r["mode"] == "gradient" and r["status"] == "ok"]
        if grad:
            out.append(f"\nGRADIENT BOARD\n{'adapter':<16}{'case':<26}"
                       f"{'median ms':>11}{'compile s':>11}  gate")
            out.append("-" * 72)
            for r in sorted(grad, key=lambda r: r["median_s"] or 0):
                out.append(f"{r['adapter']:<16}{r['case']:<26}"
                           f"{(r['median_s'] or 0) * 1e3:>11.3f}"
                           f"{(r['first_call_s'] or 0):>11.3f}  {r['grad_gate'] or '-'}")

        bad = [r for r in rs if r["status"] not in ("ok",)]
        if bad:
            out.append("\nNOT MEASURED")
            for r in sorted(bad, key=lambda r: (r["status"], r["adapter"] or "")):
                out.append(f"  {r['status']:<18}{r['adapter']:<16}{_case_label(r):<24}"
                           f"{r['config']:<16}{str(r['reason'] or '')[:70]}")

        traced = [r for r in rs if r["traced"]]
        if traced:
            out.append(f"\n  ({len(traced)} traced runs excluded from the timing table -- "
                       f"VizTracer overhead is per-Python-call and would flatter "
                       f"vectorised codes)")

    if len({k[0] for k in by_machine}) > 1:
        out.append("\nWARNING: results span multiple machines. They are reported "
                   "separately above and must not be combined -- MKL dispatch in "
                   "particular differs by CPU vendor.")
    if len(contracts) > 1:
        out.append(
            f"\nWARNING: results span measurement contracts {sorted(contracts)}. "
            "primitive-v1 timed each library's transform entry point; idiomatic-v1 "
            "times the call its documentation puts in front of a user, which for "
            "POPPY is 83% slower at N=1024. The older rows are stale, not a "
            "regression -- re-run them or delete them.")
    return "\n".join(out)


def amortisation(rows: list[dict], case: str, config: str) -> dict[str, list[tuple[int, float]]]:
    """Total time vs number of propagations: T(k) = build + first_call + k*steady.

    The most decision-useful curve in the suite. The lines cross, and where they
    cross tells a user which code to reach for at their actual workload -- which
    is the question a bar chart of steady-state time cannot answer.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r["case"] != case or r["config"] != config or r["status"] != "ok":
            continue
        if not r["median_s"]:
            continue
        # Scan points are keyed per size: build cost grows with N too, so the
        # crossover moves, and collapsing the sizes onto one adapter key would
        # keep whichever point happened to be read last.
        key = r["adapter"] if r.get("scan_value") is None \
            else f"{r['adapter']}@{r['scan_value']}"
        setup = (r["build_s"] or 0.0) + (r["first_call_s"] or 0.0)
        out[key] = [(k, setup + k * r["median_s"]) for k in (1, 10, 100, 1000, 10000)]
    return out
