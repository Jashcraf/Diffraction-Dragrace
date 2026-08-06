"""Chrome/Perfetto trace -> self-time by category.

This is what turns a flame graph into a comparable number. Self time is
aggregated per function and mapped to categories through a per-adapter regex
table (adapters/<name>/categories.yaml), producing a breakdown like

    fft 61%   elementwise 17%   alloc 9%   units 8%   python_overhead 5%

across every code on the same case. That comparison -- how much of a
pupil-to-focus propagation is actually the transform, versus unit handling,
array copies, coordinate regeneration and interpreter dispatch -- is closer to
"does this code implement the physics efficiently" than wall time is, and it
survives being run on a different machine.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

DEFAULT_CATEGORIES = {
    "fft": [r"\.fft\.", r"pyfftw", r"mkl_fft", r"cufft", r"scipy\.fft"],
    "gemm": [r"matmul", r"\.dot\b", r"tensordot", r"einsum", r"mdft", r"_dft\b"],
    "elementwise": [r"\bexp\b", r"\bmultiply\b", r"\babs\b", r"\bangle\b", r"phasor"],
    "alloc": [r"\bzeros\b", r"\bempty\b", r"\bones\b", r"asarray", r"\bcopy\b", r"pad"],
    "units": [r"astropy\.units", r"\bQuantity\b", r"\bto_value\b"],
    "coords": [r"coordinate", r"meshgrid", r"\bgrid\b", r"fftfreq"],
}


def load_trace(path: str | Path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def self_times(trace: dict) -> dict[str, float]:
    """Self time per function name, in seconds.

    Complete ('X') events carry a duration that includes their children, so a
    per-thread stack subtracts each frame's children from its total. Events are
    sorted by (start, -duration) so that a parent is always opened before a
    child that begins at the same timestamp.
    """
    events = [e for e in trace.get("traceEvents", [])
              if e.get("ph") == "X" and "dur" in e and "ts" in e]
    by_thread: dict[tuple, list] = defaultdict(list)
    for e in events:
        by_thread[(e.get("pid"), e.get("tid"))].append(e)

    out: dict[str, float] = defaultdict(float)
    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack: list[dict] = []
        for e in evs:
            ts, dur = float(e["ts"]), float(e["dur"])
            while stack and ts >= stack[-1]["end"]:
                f = stack.pop()
                out[f["name"]] += (f["dur"] - f["children"])
            if stack:
                stack[-1]["children"] += dur
            stack.append({"end": ts + dur, "name": e["name"], "dur": dur, "children": 0.0})
        while stack:
            f = stack.pop()
            out[f["name"]] += (f["dur"] - f["children"])
    return {k: v / 1e6 for k, v in out.items()}         # microseconds -> seconds


def load_categories(adapter_dir: str | Path) -> dict[str, list[str]]:
    p = Path(adapter_dir) / "categories.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text()) or DEFAULT_CATEGORIES
    return DEFAULT_CATEGORIES


def categorise(times: dict[str, float], categories: dict[str, list[str]]) -> dict[str, float]:
    compiled = {c: [re.compile(p) for p in pats] for c, pats in categories.items()}
    out: dict[str, float] = defaultdict(float)
    for name, t in times.items():
        for cat, pats in compiled.items():
            if any(p.search(name) for p in pats):
                out[cat] += t
                break
        else:
            out["python_overhead"] += t
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def summarise(trace_path: str | Path, adapter_dir: str | Path) -> dict:
    trace = load_trace(trace_path)
    times = self_times(trace)
    cats = categorise(times, load_categories(adapter_dir))
    top = sorted(times.items(), key=lambda kv: -kv[1])[:20]
    return {
        "self_time_frac": cats,
        "total_self_time_s": sum(times.values()),
        "top_functions": [{"name": n, "self_s": t} for n, t in top],
    }
