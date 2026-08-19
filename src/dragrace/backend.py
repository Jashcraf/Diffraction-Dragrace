"""Backend resolution and verification.

The premise: never trust that a config was honoured. HCIPy's FFT layer probes
for mkl_fft and pyfftw at import and uses them if present; POPPY has accel_math
toggles that fall back silently; prysm swaps its entire array module. Install
mkl_fft into a shared environment and half the "NumPy baseline" runs quietly
become MKL runs, and nothing in the output says so.

So every adapter reports what it actually resolved, and the worker refuses to
record a result whose resolved backend contradicts the requested one.
"""
from __future__ import annotations

from typing import Any

from .config import Config


def threadpool_snapshot() -> list[dict[str, Any]]:
    """threadpoolctl's view: library paths on disk, not package metadata.

    The authoritative answer to "which BLAS am I really running, with how many
    threads". Recorded verbatim in every result.
    """
    try:
        import threadpoolctl
        return [dict(p) for p in threadpoolctl.threadpool_info()]
    except Exception as exc:                          # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def detect_blas() -> str:
    for p in threadpool_snapshot():
        api = str(p.get("internal_api", "")).lower()
        if "mkl" in api:
            return "mkl"
        if "openblas" in api:
            return "openblas"
        if "accelerate" in api or "vecLib" in str(p.get("filepath", "")):
            return "accelerate"
    return "unknown"


def detect_thread_counts() -> list[dict]:
    """One entry per loaded threading runtime.

    Returned as a list, not a dict keyed by internal_api: a process routinely
    has more than one OpenBLAS or OpenMP runtime loaded (NumPy's and SciPy's,
    for instance) with *different* thread counts. Collapsing them into a dict
    keeps only the last, which silently masks exactly the mismatch this
    function exists to detect.
    """
    return [
        {"api": str(p.get("internal_api", f"lib{i}")),
         "threads": int(p.get("num_threads", -1)),
         "path": str(p.get("filepath", ""))}
        for i, p in enumerate(threadpool_snapshot())
        if "num_threads" in p
    ]


def numpy_fft_module() -> str:
    """Which module numpy.fft.fft2 actually lives in.

    Catches monkeypatched or shimmed FFT front ends, including the harness's
    own ledger instrumentation, which must never be active during a timing run.
    """
    try:
        import numpy as np
        fn = np.fft.fft2
        return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
    except Exception as exc:                          # noqa: BLE001
        return f"error: {exc}"


def available_fft_backends() -> dict[str, bool]:
    out = {}
    for name, mod in (("mkl", "mkl_fft"), ("pyfftw", "pyfftw"), ("scipy_pocketfft", "scipy.fft")):
        try:
            __import__(mod)
            out[name] = True
        except Exception:                             # noqa: BLE001
            out[name] = False
    out["numpy"] = True
    return out


class BackendMismatch(RuntimeError):
    pass


def verify(config: Config, resolved: dict[str, Any], strict: bool = True,
           not_selectable: tuple[str, ...] = ()) -> list[str]:
    """Compare requested against resolved. Returns warnings; raises on conflict.

    A mismatch is a hard failure rather than a warning because a mislabelled
    result is worse than no result: it survives into the report, gets plotted,
    and nobody can tell by looking.

    `not_selectable` names axes the adapter has declared it cannot honour --
    dLux supplies its own XLA kernels and has no FFT or BLAS to choose. Those
    axes are downgraded from failure to a loud warning rather than exempted
    quietly: the run is legitimate (it is the only backend that code has), but
    it does NOT belong in a backend comparison, and every result says so.
    """
    problems: list[str] = []
    warnings: list[str] = []

    got_blas = resolved.get("blas") or detect_blas()
    if config.blas_backend != "unknown" and got_blas != "unknown":
        if got_blas != config.blas_backend:
            msg = f"BLAS: requested {config.blas_backend!r}, threadpoolctl reports {got_blas!r}"
            (warnings if "blas" in not_selectable else problems).append(msg)
    if "blas" in not_selectable:
        warnings.append(
            f"BLAS axis is not selectable for this adapter; the config's "
            f"{config.blas_backend!r} did not apply and this row is not comparable "
            f"along the BLAS axis."
        )

    got_fft = resolved.get("fft_backend")
    if got_fft and got_fft != config.fft_backend:
        # 'native' and 'xla' mean "whatever the device path uses" and are not
        # expected to match a named CPU library.
        if config.fft_backend not in ("native", "xla"):
            msg = f"FFT: requested {config.fft_backend!r}, adapter resolved {got_fft!r}"
            if "fft" in not_selectable:
                warnings.append(
                    msg + " -- this adapter has no selectable FFT backend, so the "
                          "config's FFT axis did not apply to this run."
                )
            else:
                problems.append(msg)

    got_device = str(resolved.get("device", "")).lower()
    if got_device:
        want_gpu = config.is_gpu
        has_gpu = "cuda" in got_device or "gpu" in got_device
        if want_gpu != has_gpu:
            problems.append(
                f"device: requested {config.device!r}, adapter resolved {got_device!r}"
            )

    if "threads" in not_selectable:
        # threadpoolctl only sees OpenMP/BLAS runtimes. An XLA-backed run has
        # neither in its path, so a "1 thread" reading there describes NumPy
        # sitting idle in the same process and says nothing about the code being
        # measured. Recorded as unverified rather than left to look checked:
        # XLA_FLAGS=--xla_cpu_multi_thread_eigen/intra_op_parallelism_threads
        # produced identical timings at 1 and 8 threads when this was tested, so
        # the harness cannot currently prove the thread count either way.
        warnings.append(
            f"thread axis is not verifiable for this adapter: it runs on XLA, "
            f"which threadpoolctl cannot see. The config's threads={config.threads} "
            f"is a label, not a measured property, and this row must not be used "
            f"for a thread-scaling comparison."
        )

    # A run labelled threads=1 that actually used 24 is a mislabelled result, so
    # a mismatch on the *active* BLAS is fatal. Other loaded runtimes (an idle
    # MKL alongside an active OpenBLAS, say) only warn -- their thread count
    # does not affect the measurement.
    active = got_blas
    for e in detect_thread_counts():
        if e["threads"] == config.threads:
            continue
        msg = (f"thread count: {e['api']} reports {e['threads']} threads, config asked "
               f"for {config.threads}  [{e['path']}]")
        if e["api"] == active and "threads" not in not_selectable:
            problems.append(
                msg + "\n    The active BLAS is not honouring the requested thread count. "
                      "This usually means NumPy was imported before the config's "
                      "environment variables were applied -- OpenBLAS and MKL read them "
                      "at load time."
            )
        else:
            warnings.append(msg)

    if problems and strict:
        raise BackendMismatch(
            "resolved backend does not match the requested config:\n  - "
            + "\n  - ".join(problems)
            + "\n\nThis run would produce a mislabelled result, so it was refused. "
              "Check that this adapter is running in the environment that serves "
              f"config {config.id!r} (see envs/README.md)."
        )
    return warnings + problems


def snapshot(config: Config, resolved: dict[str, Any],
             not_selectable: tuple[str, ...] = ()) -> dict[str, Any]:
    """The `backend` block recorded in every result.json."""
    return {
        # Axes the adapter declared it cannot honour. Recorded at the top of the
        # block because it changes how every other field in it should be read.
        "axes_not_selectable": list(not_selectable),
        "requested": {
            "fft": config.fft_backend,
            "blas": config.blas_backend,
            "threads": config.threads,
            "device": config.device,
            "precision_override": config.precision_override,
        },
        "resolved": dict(resolved),
        "detected": {
            "blas": detect_blas(),
            "thread_counts": detect_thread_counts(),
            "numpy_fft_module": numpy_fft_module(),
            "fft_backends_importable": available_fft_backends(),
        },
        "threadpool_info": threadpool_snapshot(),
    }
