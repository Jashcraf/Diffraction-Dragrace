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
    #: Per-adapter overrides of `conda_env`, for configs one environment cannot
    #: serve. The GPU configs are the case that forces this: pip's jax[cuda12]
    #: wheels vendor their own CUDA runtime and must not share an interpreter
    #: with conda's CUDA runtime for CuPy, so `gpu_f64` needs
    #: dragrace-gpu-cupy for poppy/prysm and dragrace-gpu-jax for dLux. Two
    #: config ids would be the wrong fix: `config` names the *machine
    #: configuration* -- device, precision, FFT, BLAS -- and splitting it by
    #: which Python packaging accident serves an adapter would put dLux-on-CUDA
    #: and prysm-on-CUDA on separate boards, which is exactly the comparison the
    #: GPU configs exist to make.
    conda_env_by_adapter: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def env_for(self, adapter: str | None) -> str | None:
        """The conda environment that serves `adapter` under this config."""
        if adapter is None:
            return self.conda_env
        return self.conda_env_by_adapter.get(adapter, self.conda_env)

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

        Threads: NOT SET HERE, because on this jaxlib no XLA_FLAGS setting can
        do it. This block used to emit

            --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=N

        and both halves were inert. MEASURED on jaxlib 0.10.2, dLux at
        N_p=1024: cpu/wall = 10.06 with no XLA_FLAGS at all, 10.04 with the
        eigen flag, 9.92 with the full string -- i.e. ~10 cores in every case,
        on a config labelled threads=1. `--xla_cpu_multi_thread_eigen` governed
        the old Eigen path that the thunk runtime replaced, and
        `--xla_cpu_use_thunk_runtime=false` does not bring it back (10.05).
        `intra_op_parallelism_threads` is not an XLA flag at all: it is a
        TensorFlow session option, and the only reason it never crashed the
        worker is that it lacked the `--` prefix and was discarded as a
        positional. Adding the prefix aborts the process --
        "Unknown flag in XLA_FLAGS" -- so the string was one plausible-looking
        cleanup away from taking every JAX run down with it.

        What works is CPU affinity, applied by the worker before any import
        (dragrace.worker.main). That is a property of the process rather than of
        the library, so it constrains XLA's thread pool, OpenBLAS, MKL and
        anything else that sizes itself from the core count -- and it is
        verified after the fact by the cpu/wall ratio recorded in every timing
        block, so a threads=1 row that used ten cores can no longer be published
        as one that did not.
        """
        return {"JAX_ENABLE_X64": "0" if self.precision_override == "complex64" else "1"}

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
        d["conda_env_by_adapter"] = {
            str(k): str(v) for k, v in (d.get("conda_env_by_adapter") or {}).items()}
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
