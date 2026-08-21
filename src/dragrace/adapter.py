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
    #: Where this code's grids sit: "pixel" (origin on sample N//2, the
    #: fftshift convention) or "interpixel" (origin between the middle two
    #: samples). Declared rather than imposed, because several libraries fix it
    #: internally with no documented knob -- POPPY's OpticalSystem is
    #: interpixel and cannot be talked out of it. The reference field, the
    #: injected pupil and the coordinate grids are all built to match, so a code
    #: is gated on the physics rather than on whose convention it adopted.
    #: See dragrace.grid and docs/conventions.md.
    grid_centering: str = "pixel"
    #: What this code's documented entry point returns, and therefore what the
    #: accuracy gate can honestly check: "field" (default) or "intensity".
    #: PROPER's prop_end returns intensity unless asked otherwise, and its focal
    #: phase carries a residual quadratic curvature from propagating through a
    #: lens -- so its PSF is right to ~1e-7 while its field fails a field gate.
    #: Declaring "intensity" gates |E|^2 and marks the row: conjugation and
    #: scale phase come back null, and it must not feed a phase-sensitive claim.
    output_quantity: str = "field"
    #: Importable modules this adapter needs. Checked before supports(), so a
    #: library that simply is not installed reports `unsupported` with a clear
    #: reason rather than failing somewhere inside build() -- "not installed"
    #: and "broken" are different findings and the report must distinguish them.
    requires: tuple[str, ...] = ()

    def check_requirements(self, deep: bool = True) -> Supported:
        """Actually import, rather than just find_spec.

        find_spec only proves a module is discoverable on disk. A broken install
        -- a library present but whose own dependencies fail to initialise --
        would pass that check and then surface later as an opaque
        "ImportError: initialization failed" from inside build(). Importing here
        costs nothing extra (the worker is about to import it anyway) and lets
        the three cases be distinguished in the report: not installed, installed
        but broken, and working.

        `deep=False` stops at find_spec, and exists for one caller: the ledger
        pass. Several libraries bind NumPy's entry points into closures while
        being imported -- hcipy/_math/fft.py captures `getattr(np.fft, name)` at
        module scope -- so a library imported before the instrumentation goes on
        can never be intercepted, and the ledger records a silent zero for a
        propagation that really ran two FFTs. Deferring the import into the
        patched region is the only way to price those codes. The cost is that a
        broken install then surfaces as a traceback from inside build() rather
        than as a clean `unsupported`, which is an acceptable trade for a
        diagnostic mode and no trade at all for the timing modes.
        """
        for m in self.requires:
            if importlib.util.find_spec(m) is None:
                return Unsupported(
                    f"not installed: {m}. See envs/README.md for which environment "
                    f"provides it."
                )
            if not deep:
                continue
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
        """Materialise `result` on the host, as the library hands it to a user.

        Timed separately from compute. This is whatever the documented call
        returns -- for a code whose user-facing entry point returns intensity,
        that is an intensity array, and the accuracy gate reads
        `complex_field()` instead.
        """
        return np.asarray(result)

    def complex_field(self, state: Any, result: Any) -> np.ndarray:
        """Complex focal field for the accuracy gate. NOT timed.

        Defaults to to_host(), which is right for every code whose documented
        propagation returns a field. It exists for the codes whose documented
        entry point returns an *intensity* PSF -- POPPY's `calc_psf` returns a
        FITS HDUList -- where gating on intensity alone would throw away the
        phase-sign and normalisation diagnostics that make a cross-code
        comparison meaningful.

        Overriding this is not a licence to compute the answer differently: it
        must be the same propagation, obtained through whatever the library
        documents for recovering the field (POPPY: `calc_psf(return_final=True)`).
        The timed path stays exactly what a user would call, so the extra cost
        of asking for the field never lands in the measurement.
        """
        return self.to_host(result)

    def teardown(self, state: Any) -> None:
        return None

    # ----------------------------------------------- phase-retrieval board --
    #: Which phase-retrieval board this adapter belongs on, or None if it has no
    #: retrieval forward model here. "numerical" means it supplies a scalar loss
    #: and the optimiser forms its own finite-difference gradient; "analytic"
    #: means it returns (loss, dloss/dtheta). An adapter appears on exactly one
    #: of the two boards, matched against the case's `retrieval.gradient`, so
    #: that a figure never mixes a code costing P+1 forward models per gradient
    #: with one costing O(1). A tuple declares both, which only the NumPy
    #: baseline uses -- it is the floor for either board rather than a
    #: competitor on one.
    retrieval_gradient: str | tuple[str, ...] | None = None

    #: Devices this adapter's retrieval forward model actually runs on. Declared
    #: per adapter rather than gated globally: supporting a propagation on the
    #: GPU does not imply supporting a *retrieval* there, because the optimiser
    #: is host-side and every loss and gradient has to cross the boundary. An
    #: adapter that leaves the chain in host NumPy would otherwise be labelled
    #: `gpu_f64` while computing on the CPU -- a silently wrong row rather than
    #: a missing one.
    retrieval_devices: tuple[str, ...] = ("cpu",)

    def retrieval_support(self, case: Case, config: Config) -> Supported:
        """Shared gate for kind=phase_retrieval. Call from supports().

        Kept on the base class so that seven adapters cannot drift on what
        "belongs on this board" means, and so the refusal reads the same way in
        every result file.
        """
        if self.retrieval_gradient is None:
            return Unsupported(
                f"{self.name} has no phase-retrieval forward model in this harness. "
                f"That is a statement about the adapter, not about the library."
            )
        want = case.retrieval.gradient
        allowed = ((self.retrieval_gradient,)
                   if isinstance(self.retrieval_gradient, str)
                   else tuple(self.retrieval_gradient))
        if want not in allowed:
            return Unsupported(
                f"{self.name} supplies a {'/'.join(allowed)} gradient and this "
                f"case asks for a {want} one. The two boards are separate on purpose: "
                f"a finite-difference gradient costs P+1 forward models and an "
                f"analytic one costs O(1), so putting them on one axis would show a "
                f"difference in differentiation method as a difference in propagation "
                f"speed. See cases/phase_retrieval/ and docs/phase_retrieval_board.md."
            )
        if config.is_gpu and "gpu" not in self.retrieval_devices:
            return Unsupported(
                f"{self.name} has no GPU phase-retrieval path in this harness: its "
                f"forward model is written in host NumPy, so running it under "
                f"{config.id} would produce a row labelled with a device it never "
                f"touched. See Adapter.retrieval_devices.")
        return True

    def retrieval_psf(self, state: Any, theta: np.ndarray) -> np.ndarray:
        """This adapter's own forward model at `theta`, as an intensity PSF.

        NOT timed, and not part of the retrieval. It exists so the harness can
        ask a separate question from "how fast did it converge": does this code
        model the telescope the case describes at all? Each adapter fits its own
        observed PSF -- generated by this same forward model at the truth
        coefficients -- so a code with a subtly wrong pupil would converge
        beautifully onto its own private physics and nothing in the timing would
        notice. The worker compares this against
        dragrace.retrieval.reference_psf and records the result as
        `forward_accuracy`.
        """
        raise NotImplementedError

    def retrieval_report(self, state: Any, result: Any) -> dict:
        """Untimed diagnostics for one retrieval: iterations, evaluations, loss.

        Runtime alone is not readable on this board -- two codes can differ both
        in what one forward model costs and in how many the optimiser wanted,
        and only the product is the wall time. Returning the counts is what lets
        the report divide one by the other.
        """
        return {}

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
