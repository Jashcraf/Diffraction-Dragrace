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


def interpreter_for(config: Config, allow_fallback: bool = False,
                    adapter: str | None = None) -> tuple[str | None, str]:
    """(python executable, note). None means "skip this run".

    `adapter` selects among the config's per-adapter environment overrides. The
    GPU configs need it: one config id is served by two environments, because
    pip's CUDA JAX and conda's CuPy cannot share an interpreter.
    """
    env_name = config.env_for(adapter)
    if not env_name:
        return sys.executable, "current interpreter (config names no env)"

    root = conda_root()
    if root is not None:
        cand = root / "envs" / env_name / "bin" / "python"
        if cand.exists():
            return str(cand), f"conda env {env_name}"
        if root.name == env_name and (root / "bin" / "python").exists():
            return str(root / "bin" / "python"), f"conda base ({env_name})"

    if allow_fallback:
        return sys.executable, (f"FALLBACK: env {env_name!r} not found, using the "
                                f"active interpreter -- results are NOT attributable")
    return None, (f"environment {env_name!r} not found; create it with "
                  f"`conda env create -f envs/...` (see envs/README.md)")


def conda_env_vars(python: str) -> dict[str, str]:
    """Make a spawned worker's environment name the env it is actually running in.

    Workers are launched as `$PREFIX/bin/python`, never through
    `conda activate`, so the inherited CONDA_PREFIX still names whichever
    environment the SWEEP was started from. Most libraries do not care. CuPy
    does: cupy._environment._get_cuda_path() consults CONDA_PREFIX to locate the
    toolkit, and when that points at an environment with no CUDA in it the next
    candidate is /usr/local/cuda -- the system install, which has no reason to
    match the driver.

    MEASURED, because the failure is not a subtle skew but a hard stop that
    reads like a broken GPU. On this machine the system toolkit is CUDA 13.1 and
    the driver is 12.8, so NVRTC builds an image the driver refuses and every
    CuPy kernel dies with `CUDA_ERROR_INVALID_IMAGE: device kernel image is
    invalid` -- surfacing from a plain `cupy.arange` inside prysm's
    prepare_executor, which reads like a prysm bug and is not one. Both poppy
    and prysm failed that way across the whole phase-retrieval board when the
    sweep was launched from the CPU environment, and both passed when it was
    launched from `dragrace-gpu-cupy`: the same run, differing only in an
    inherited variable. `conda_env` in a config is a promise about where a
    result was computed, and this is what keeps it from being half-kept.

    PATH gets the env's bin ahead of the inherited one for the same reason: a
    worker that shells out should find its own environment's tools first.
    """
    prefix = Path(python).parent.parent
    if not (prefix / "conda-meta").is_dir():
        return {}                    # not a conda env; leave the caller's alone
    return {
        "CONDA_PREFIX": str(prefix),
        "CONDA_DEFAULT_ENV": prefix.name,
        "PATH": str(prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
    }


def cuda_path_for(python: str) -> str | None:
    """CUDA_PATH for a conda interpreter whose CUDA came from conda-forge.

    CuPy compiles its elementwise kernels with NVRTC at first use and needs the
    toolkit *headers* on disk to do it. conda-forge puts them under
    `$PREFIX/targets/<arch>/`, not `$PREFIX/include`, and CuPy appends
    `/include` to whatever CUDA_PATH says -- so pointing CUDA_PATH at the
    prefix, which is the obvious guess, resolves to a directory that does not
    exist and every kernel launch dies with "Failed to find CUDA headers".

    This has to be set by the runner rather than by `conda activate`: workers
    are launched as `$PREFIX/bin/python`, so an environment's activate.d hooks
    never run. Returns None when the interpreter has no conda CUDA, leaving any
    inherited CUDA_PATH alone.
    """
    prefix = Path(python).parent.parent
    for target in sorted((prefix / "targets").glob("*")) if (prefix / "targets").is_dir() else []:
        if (target / "include" / "cuda_runtime.h").exists():
            return str(target)
    return None


def result_dir(root: Path, machine_id: str, run_id: str, spec: RunSpec) -> Path:
    return (root / "raw" / machine_id.replace(":", "_") / run_id /
            spec.adapter / spec.config.id / spec.case.id / spec.mode)


def run_one(spec: RunSpec, results_root: Path, run_id: str, machine_id: str,
            allow_fallback: bool = False, strict_backend: bool = True,
            repo_root: Path | None = None) -> dict:
    repo_root = repo_root or Path.cwd()
    out = result_dir(results_root, machine_id, run_id, spec)
    out.mkdir(parents=True, exist_ok=True)

    python, note = interpreter_for(spec.config, allow_fallback, spec.adapter)
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
    # Before the CUDA block below, because CuPy resolves its toolkit from
    # CONDA_PREFIX when CUDA_PATH is unset and a worker inheriting the launching
    # environment's prefix silently reaches for the system CUDA instead.
    env.update(conda_env_vars(python))
    cuda = cuda_path_for(python)
    if cuda:
        # CUDA_HOME too: cuda.pathfinder warns when the two disagree, and a
        # stale /usr/local/cuda inherited from the shell is the usual source of
        # the disagreement.
        env["CUDA_PATH"] = env["CUDA_HOME"] = cuda
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
