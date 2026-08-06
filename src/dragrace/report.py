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


def aggregate(results_root: str | Path) -> list[dict]:
    rows = []
    for p in sorted(Path(results_root).rglob("result.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:                              # noqa: BLE001
            continue
        t = r.get("timing", {}) or {}
        stats = t.get("device_compute_stats", {}) or {}
        acc = r.get("accuracy", {}) or {}
        fl = r.get("flops", {}) or {}
        rows.append({
            "adapter": (r.get("adapter") or {}).get("name"),
            "case": r.get("case_id"),
            "config": r.get("config_id"),
            "mode": r.get("mode"),
            "status": r.get("status"),
            "machine": (r.get("machine") or {}).get("id"),
            "cpu": (r.get("machine") or {}).get("cpu"),
            "median_s": stats.get("median"),
            "min_s": stats.get("min"),
            "iqr_s": stats.get("iqr"),
            "traced": t.get("traced", False),
            "rel_l2": acc.get("rel_l2"),
            "gate": acc.get("gate"),
            "ideal_gflop": (fl.get("ideal") or {}).get("flops", 0) / 1e9 or None,
            "ledger_gflop": (fl.get("ledger") or {}).get("flops_total", 0) / 1e9 or None,
            "A_overhead": fl.get("algorithmic_overhead"),
            "build_s": (r.get("setup") or {}).get("build_s"),
            "first_call_s": (r.get("setup") or {}).get("first_call_s"),
            "mem_peak_mib": ((r.get("memory") or {}).get("tracemalloc_peak_bytes") or 0) / 2**20
                            or None,
            "grad_gate": (r.get("gradient_accuracy") or {}).get("gate"),
            "reason": r.get("reason"),
            "path": str(p),
        })
    return rows


def render_text(rows: list[dict]) -> str:
    by_machine = defaultdict(list)
    for r in rows:
        by_machine[(r["machine"], r["cpu"])].append(r)

    out = []
    for (mid, cpu), rs in by_machine.items():
        out.append(f"\n=== machine {mid}  {cpu} ===")
        timed = [r for r in rs if r["status"] == "ok" and r["median_s"] and not r["traced"]]
        if timed:
            out.append(f"\n{'adapter':<16}{'case':<24}{'config':<16}"
                       f"{'median ms':>11}{'GFLOP':>9}{'GFLOP/s':>10}{'A':>7}  gate")
            out.append("-" * 100)
            for r in sorted(timed, key=lambda r: (r["case"], r["config"], r["median_s"])):
                g = r["ideal_gflop"] or 0
                rate = g / r["median_s"] if r["median_s"] else 0
                a = f"{r['A_overhead']:.2f}" if r["A_overhead"] else "-"
                out.append(f"{r['adapter']:<16}{r['case']:<24}{r['config']:<16}"
                           f"{r['median_s'] * 1e3:>11.3f}{g:>9.3f}{rate:>10.2f}{a:>7}  "
                           f"{r['gate'] or '-'}")

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
                out.append(f"  {r['status']:<18}{r['adapter']:<16}{r['case']:<24}"
                           f"{r['config']:<16}{str(r['reason'] or '')[:70]}")

        traced = [r for r in rs if r["traced"]]
        if traced:
            out.append(f"\n  ({len(traced)} traced runs excluded from the timing table -- "
                       f"VizTracer overhead is per-Python-call and would flatter "
                       f"vectorised codes)")

    if len(by_machine) > 1:
        out.append("\nWARNING: results span multiple machines. They are reported "
                   "separately above and must not be combined -- MKL dispatch in "
                   "particular differs by CPU vendor.")
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
        setup = (r["build_s"] or 0.0) + (r["first_call_s"] or 0.0)
        out[r["adapter"]] = [(k, setup + k * r["median_s"])
                             for k in (1, 10, 100, 1000, 10000)]
    return out
