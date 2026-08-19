"""Orchestration: spawns one worker per (adapter x case x config x mode).

Each run gets its own interpreter, chosen from the config's `conda_env`. That is
what lets a single sweep span environments -- the OpenBLAS env, the MKL env and
the two CUDA envs -- which is necessary because NumPy links exactly one BLAS and
because pip's CUDA JAX and conda's CuPy cannot share an interpreter.

If the environment named by a config does not exist, the run is skipped with a
recorded reason rather than silently executed in whatever interpreter happens
to be active. Running the MKL config in the OpenBLAS environment would produce
a result labelled `cpu_mkl_1t` containing OpenBLAS numbers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .case import Case
from .config import Config


@dataclass
class RunSpec:
    case: Case
    config: Config
    adapter: str
    mode: str
    case_file: str          # workers re-parse the YAML rather than receive a
    config_file: str        # pickled object, so a run is reproducible from argv


def conda_root() -> Path | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        return p.parent.parent if p.parent.name == "envs" else p
    exe = os.environ.get("CONDA_EXE")
    if exe:
        return Path(exe).parent.parent
    for cand in (Path.home() / "anaconda3", Path.home() / "miniconda3",
                 Path.home() / "miniforge3"):
        if cand.exists():
            return cand
    return None


def interpreter_for(config: Config, allow_fallback: bool = False) -> tuple[str | None, str]:
    """(python executable, note). None means "skip this run"."""
    if not config.conda_env:
        return sys.executable, "current interpreter (config names no env)"

    root = conda_root()
    if root is not None:
        cand = root / "envs" / config.conda_env / "bin" / "python"
        if cand.exists():
            return str(cand), f"conda env {config.conda_env}"
        if root.name == config.conda_env and (root / "bin" / "python").exists():
            return str(root / "bin" / "python"), f"conda base ({config.conda_env})"

    if allow_fallback:
        return sys.executable, (f"FALLBACK: env {config.conda_env!r} not found, using the "
                                f"active interpreter -- results are NOT attributable")
    return None, (f"environment {config.conda_env!r} not found; create it with "
                  f"`conda env create -f envs/...` (see envs/README.md)")


def result_dir(root: Path, machine_id: str, run_id: str, spec: RunSpec) -> Path:
    return (root / "raw" / machine_id.replace(":", "_") / run_id /
            spec.adapter / spec.config.id / spec.case.id / spec.mode)


def run_one(spec: RunSpec, results_root: Path, run_id: str, machine_id: str,
            allow_fallback: bool = False, strict_backend: bool = True,
            repo_root: Path | None = None) -> dict:
    repo_root = repo_root or Path.cwd()
    out = result_dir(results_root, machine_id, run_id, spec)
    out.mkdir(parents=True, exist_ok=True)

    python, note = interpreter_for(spec.config, allow_fallback)
    if python is None:
        res = {"status": "skipped", "reason": note, "case_id": spec.case.id,
               "config_id": spec.config.id, "adapter": {"name": spec.adapter},
               "mode": spec.mode}
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res

    cmd = [python, "-m", "dragrace.worker",
           "--case", spec.case_file, "--config", spec.config_file,
           "--adapter", spec.adapter, "--mode", spec.mode, "--out", str(out)]
    if not strict_backend:
        cmd.append("--no-strict-backend")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(spec.config.full_env())

    # Preflight: is the harness even importable there? Without this the failure
    # surfaces as a runpy traceback about a missing transitive dependency, which
    # reads like a harness bug rather than "that environment is not provisioned".
    probe = subprocess.run([python, "-c", "import dragrace"], cwd=repo_root, env=env,
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        missing = probe.stderr.strip().splitlines()[-1] if probe.stderr else "unknown error"
        res = {"status": "skipped", "case_id": spec.case.id, "config_id": spec.config.id,
               "adapter": {"name": spec.adapter}, "mode": spec.mode,
               "reason": (f"conda environment {spec.config.conda_env!r} does not have the "
                          f"harness installed ({missing}). Create it from environment.yml "
                          f"/ envs/, then `pip install -e .` -- see envs/README.md.")}
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res

    # timeout_s is per measurement, so a scan case gets it once per point --
    # otherwise adding sizes to a scan would silently start killing it.
    proc = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True,
                          timeout=spec.case.total_timeout_s + 120)
    rpath = out / "result.json"
    if rpath.exists():
        res = json.loads(rpath.read_text())
    else:
        res = {"status": "failed", "reason": proc.stderr.strip()[-2000:] or "no result written",
               "case_id": spec.case.id, "config_id": spec.config.id,
               "adapter": {"name": spec.adapter}, "mode": spec.mode}
        rpath.write_text(json.dumps(res, indent=2))
    res["_interpreter_note"] = note
    return res


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
