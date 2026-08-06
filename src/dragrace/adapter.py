"""The adapter contract -- the only thing a contributor writes.

Four methods carry the whole design:

  build()      untimed. Grids, matrices, FFT plans, JIT compilation, staging
               inputs on-device. Anything a real user would hoist out of a loop.
  propagate()  timed. May return handles that are not yet computed.
  sync()       blocks until the result physically exists. No-op on NumPy,
               cp.cuda.Stream.null.synchronize() on CuPy, block_until_ready on
               JAX. Called INSIDE the clock.
  to_host()    device -> host copy, timed and reported separately.

sync() is the reason JAX's asynchronous dispatch and CuPy's asynchronous kernel
launches do not silently produce 100x speedups: without it, propagate() returns
before any arithmetic has happened. See docs/methodology.md.
"""
from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .case import Case
from .config import Config


@dataclass(frozen=True)
class Unsupported:
    """Returned rather than raised, so the report can render an honest matrix
    of *why* each hole exists instead of a blank cell."""

    reason: str

    def __bool__(self) -> bool:
        return False


Supported = bool | Unsupported


class Adapter(ABC):
    name: str = "unnamed"
    #: "verified" once exercised against the real library on this machine;
    #: "unverified" while the API calls are written from documentation.
    #: `dragrace doctor` surfaces this so nobody mistakes an untested adapter
    #: for a measured result.
    status: str = "unverified"
    #: Who has reviewed this adapter for fairness. Adapter authors are not
    #: neutral parties -- see docs/methodology.md on the hand-written-adjoint
    #: confound.
    reviewed_by: str = ""
    #: Importable modules this adapter needs. Checked before supports(), so a
    #: library that simply is not installed reports `unsupported` with a clear
    #: reason rather than failing somewhere inside build() -- "not installed"
    #: and "broken" are different findings and the report must distinguish them.
    requires: tuple[str, ...] = ()

    def check_requirements(self) -> Supported:
        """Actually import, rather than just find_spec.

        find_spec only proves a module is discoverable on disk. A broken install
        -- a library present but whose own dependencies fail to initialise --
        would pass that check and then surface later as an opaque
        "ImportError: initialization failed" from inside build(). Importing here
        costs nothing extra (the worker is about to import it anyway) and lets
        the three cases be distinguished in the report: not installed, installed
        but broken, and working.
        """
        for m in self.requires:
            if importlib.util.find_spec(m) is None:
                return Unsupported(
                    f"not installed: {m}. See envs/README.md for which environment "
                    f"provides it."
                )
            try:
                importlib.import_module(m)
            except Exception as exc:                   # noqa: BLE001
                return Unsupported(
                    f"{m} is installed but fails to import "
                    f"({type(exc).__name__}: {exc}). This is an environment problem, "
                    f"not a benchmark result -- fix the install or use the environment "
                    f"that serves this config (envs/README.md)."
                )
        return True

    # ------------------------------------------------------------ metadata --
    @abstractmethod
    def versions(self) -> dict[str, str]:
        """Version of the library under test and anything that materially
        affects its performance."""

    def supports(self, case: Case, config: Config) -> Supported:
        return True

    def configure(self, config: Config) -> Supported:
        """Apply the config. Called after env vars are set and before build()."""
        return True

    def resolve_backend(self) -> dict[str, Any]:
        """What the library ACTUALLY resolved -- module names, device, dtype.

        Never assume the config was honoured. Several of these libraries probe
        for mkl_fft or pyfftw at import and use them silently if present, which
        is the single easiest way to produce a mislabelled result. The worker
        compares this against the request and fails the run on a mismatch.
        """
        return {}

    # ------------------------------------------------------------ lifecycle --
    @abstractmethod
    def build(self, case: Case, config: Config) -> Any:
        """Untimed setup. Returns opaque state passed to propagate()."""

    @abstractmethod
    def propagate(self, state: Any) -> Any:
        """The timed region. May return an unmaterialised handle."""

    def sync(self, result: Any) -> None:
        """Block until `result` physically exists. Called inside the clock."""
        return None

    def to_host(self, result: Any) -> np.ndarray:
        """Materialise as a host complex ndarray on the canonical focal grid."""
        return np.asarray(result)

    def teardown(self, state: Any) -> None:
        return None

    # ------------------------------------------------------ gradient board --
    def supports_gradient(self) -> Supported:
        return Unsupported("no gradient implementation")

    def build_gradient(self, case: Case, config: Config) -> Any:
        raise NotImplementedError

    def gradient(self, state: Any) -> Any:
        """Return (loss, grad) with grad real, shape (P,), d loss / d theta."""
        raise NotImplementedError


# ------------------------------------------------------------------ registry --
_REGISTRY: dict[str, type[Adapter]] = {}


def register(name: str):
    def deco(cls: type[Adapter]) -> type[Adapter]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def discover(root: str | Path = "adapters") -> dict[str, type[Adapter]]:
    """Import every adapters/<name>/adapter.py.

    Import failures are swallowed into a stub that reports itself unsupported:
    a missing propagator must not prevent the other five from running, and the
    reason belongs in the report rather than in a traceback.
    """
    root = Path(root)
    for mod_path in sorted(root.glob("*/adapter.py")):
        pkg = mod_path.parent.name
        if pkg.startswith("_") and pkg != "_numpy_baseline":
            continue
        mod_name = f"dragrace_adapters.{pkg}"
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:                     # noqa: BLE001
            _register_stub(pkg, f"{type(exc).__name__}: {exc}")
    return dict(_REGISTRY)


def _register_stub(pkg: str, reason: str) -> None:
    name = pkg.lstrip("_")

    @register(name)
    class _Stub(Adapter):                            # noqa: N801
        status = "import-failed"
        _reason = reason

        def versions(self):
            return {}

        def supports(self, case, config):
            return Unsupported(f"adapter failed to import: {self._reason}")

        def build(self, case, config):
            raise RuntimeError(self._reason)

        def propagate(self, state):
            raise RuntimeError(self._reason)


def get(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)
