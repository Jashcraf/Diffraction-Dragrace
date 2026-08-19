"""Execution configs: device, FFT backend, BLAS backend, threads, precision.

The third axis of the matrix. A benchmark run is (case x config x adapter):
the Case says what physics to compute, the Config says what machinery to
compute it with, and the adapter is the code under test.

Config exists as a separate object because backend selection is neither a
property of the physics nor a free choice of the library -- it is the variable
the benchmark is trying to isolate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FFT_BACKENDS = {"numpy", "scipy_pocketfft", "mkl", "pyfftw", "native", "xla"}
BLAS_BACKENDS = {"openblas", "mkl", "accelerate", "unknown"}


@dataclass(frozen=True)
class Config:
    id: str
    device: str = "cpu"                     # cpu | cuda:N
    fft_backend: str = "numpy"
    blas_backend: str = "openblas"
    threads: int = 1
    precision_override: str | None = None   # complex64 | complex128 | None
    env: dict[str, str] = field(default_factory=dict)
    conda_env: str | None = None            # which environment serves this config
    notes: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def device_index(self) -> int:
        return int(self.device.split(":")[1]) if ":" in self.device else 0

    def validate(self) -> None:
        p = []
        if self.fft_backend not in FFT_BACKENDS:
            p.append(f"fft_backend {self.fft_backend!r} not in {sorted(FFT_BACKENDS)}")
        if self.blas_backend not in BLAS_BACKENDS:
            p.append(f"blas_backend {self.blas_backend!r} not in {sorted(BLAS_BACKENDS)}")
        if self.threads < 1:
            p.append(f"threads must be >= 1, got {self.threads}")
        if self.precision_override not in (None, "complex64", "complex128"):
            p.append(f"precision_override {self.precision_override!r} invalid")
        if p:
            raise ValueError(f"config {self.id!r} invalid:\n  - " + "\n  - ".join(p))

    def thread_env(self) -> dict[str, str]:
        """Thread-count variables, applied per-run rather than baked into the
        conda environment -- otherwise every result would silently depend on
        which shell it was launched from."""
        n = str(self.threads)
        return {
            "OMP_NUM_THREADS": n,
            "MKL_NUM_THREADS": n,
            "OPENBLAS_NUM_THREADS": n,
            "NUMEXPR_NUM_THREADS": n,
            "VECLIB_MAXIMUM_THREADS": n,
        }

    def jax_env(self) -> dict[str, str]:
        """JAX/XLA settings that must exist before the first `import jax`.

        Both are the config's business rather than the adapter's, and both have
        to be environment variables because JAX reads them at import and they
        cannot be changed afterwards.

        x64: JAX defaults to float32/complex64. Benchmarking dLux at single
        precision against POPPY at double is not a comparison, so the flag
        follows the config's precision rather than whatever the shell had.

        Threads: OMP_NUM_THREADS does not reach XLA, which runs its own Eigen
        thread pool sized to the core count. Without this a run labelled
        `threads=1` could quietly use every core -- the exact mislabelling the
        backend verifier exists to prevent, arriving through a door it does not
        watch. (Measured at N_p=1024 the flag makes dLux slightly *faster*,
        16.4 ms against 18.2, so this costs nothing here; it is about the label
        being true, not about speed.)
        """
        e = {"JAX_ENABLE_X64": "0" if self.precision_override == "complex64" else "1"}
        if not self.is_gpu:
            e["XLA_FLAGS"] = (f"--xla_cpu_multi_thread_eigen={'false' if self.threads == 1 else 'true'} "
                              f"intra_op_parallelism_threads={self.threads}")
        return e

    def full_env(self) -> dict[str, str]:
        e = dict(self.thread_env())
        e.update(self.jax_env())
        e.update(self.env)          # the config's own env always wins
        return e

    def apply_to_environ(self) -> None:
        """Apply to os.environ. Must run BEFORE numpy/jax/cupy are imported;
        the worker does this in its preamble."""
        os.environ.update(self.full_env())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        d["env"] = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
        cfg = cls(**d)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        cfg = cls.from_dict(yaml.safe_load(path.read_text()))
        if cfg.id != path.stem:
            raise ValueError(f"config id {cfg.id!r} does not match filename {path.stem!r}")
        return cfg

    def summary(self) -> str:
        p = self.precision_override or "from_case"
        return (f"{self.id}: device={self.device} fft={self.fft_backend} "
                f"blas={self.blas_backend} threads={self.threads} precision={p}")


def load_configs(root: str | Path = "configs") -> dict[str, Config]:
    return {c.id: c for c in (Config.from_yaml(p) for p in sorted(Path(root).glob("*.yaml")))}
