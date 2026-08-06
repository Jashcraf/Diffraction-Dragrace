"""Machine and provenance fingerprint, recorded in every result.

`dragrace report` keys on machine_id and refuses to plot two machines on one
axis. That matters more than usual for this suite: MKL dispatches conservatively
on AMD parts, so an "OpenBLAS beats MKL" result on Zen and the same board on a
Xeon can genuinely disagree. Recording the CPU is what keeps that from becoming
a false conclusion.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:                                 # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def _cpu_vendor(model: str) -> str:
    m = model.lower()
    if "amd" in m or "ryzen" in m or "epyc" in m:
        return "amd"
    if "intel" in m or "xeon" in m or "core" in m:
        return "intel"
    if "apple" in m:
        return "apple"
    return "unknown"


def _ram_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:                                 # noqa: BLE001
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:                             # noqa: BLE001
            return None


def _git_sha(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root or Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root or Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "") if sha else "unknown"
    except Exception:                                 # noqa: BLE001
        return "unknown"


def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
        return gpus
    except Exception:                                 # noqa: BLE001
        return []


def machine() -> dict:
    model = _cpu_model()
    info = {
        "cpu": model,
        "cpu_vendor": _cpu_vendor(model),
        "logical_cores": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": _gpus(),
    }
    # Stable identifier for grouping results; deliberately excludes hostname so
    # published result sets do not leak machine names.
    ident = f"{model}|{info['logical_cores']}|{info['platform']}"
    info["id"] = "sha256:" + hashlib.sha256(ident.encode()).hexdigest()[:16]
    return info


def provenance(run_id: str = "") -> dict:
    return {
        "git_sha": _git_sha(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "argv": sys.argv[:],
    }


def numpy_build() -> dict:
    try:
        import numpy as np
        return {"version": np.__version__,
                "config": str(getattr(np, "show_config", lambda **k: "")(mode="dicts"))[:2000]}
    except Exception as exc:                          # noqa: BLE001
        return {"error": str(exc)}
