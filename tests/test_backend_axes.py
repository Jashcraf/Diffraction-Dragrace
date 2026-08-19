"""Config axes an adapter cannot honour, and the JAX settings a config must emit.

dLux has no FFT library to select and no BLAS to point at -- XLA emits its own
kernels. Refusing to run it on a NumPy config would keep the fastest code in the
suite off the board; running it silently would put an XLA row next to OpenBLAS
rows with nothing saying so. The middle path is a declaration, and these tests
pin the two halves of it: the run is allowed, and the caveat is recorded.
"""
import pytest

from dragrace import backend
from dragrace.config import Config


@pytest.fixture(autouse=True)
def _quiet_threadpool(monkeypatch):
    """Pin threadpoolctl's answer.

    verify() inspects the *running* process, which in a test run inherits
    whatever thread count pytest was launched with. Without this the thread
    check fires and masks the axis behaviour these tests are about.
    """
    monkeypatch.setattr(backend, "detect_thread_counts",
                        lambda: [{"api": "openblas", "threads": 1, "path": "libopenblas"}])


def _config(**over):
    d = {"id": "t", "device": "cpu", "fft_backend": "numpy", "blas_backend": "openblas",
         "threads": 1}
    d.update(over)
    return Config(**d)


def _resolved(**over):
    d = {"fft_backend": "xla", "blas": "openblas", "device": "cpu"}
    d.update(over)
    return d


# ------------------------------------------------------- declared-inert axes --
def test_undeclared_backend_mismatch_still_refuses():
    """The guard rail must keep working for adapters that do have the knob."""
    with pytest.raises(backend.BackendMismatch, match="FFT"):
        backend.verify(_config(), _resolved(), strict=True)


def test_declared_fft_axis_downgrades_to_a_warning():
    warnings = backend.verify(_config(), _resolved(), strict=True,
                              not_selectable=("fft",))
    assert any("no selectable FFT backend" in w for w in warnings)


def test_declared_blas_axis_is_always_flagged_even_when_it_matches():
    """openblas is *detected* for any process that imported NumPy, so a match
    proves nothing about a code that never calls BLAS."""
    warnings = backend.verify(_config(blas_backend="openblas"),
                              _resolved(blas="openblas"), strict=True,
                              not_selectable=("fft", "blas"))
    assert any("BLAS axis is not selectable" in w for w in warnings)


def test_declared_threads_axis_is_recorded_as_unverifiable():
    warnings = backend.verify(_config(), _resolved(), strict=True,
                              not_selectable=("fft", "blas", "threads"))
    assert any("thread axis is not verifiable" in w for w in warnings)


def test_snapshot_records_the_declaration():
    snap = backend.snapshot(_config(), _resolved(), ("fft", "blas"))
    assert snap["axes_not_selectable"] == ["fft", "blas"]


def test_no_declaration_means_no_caveat():
    snap = backend.snapshot(_config(fft_backend="xla"), _resolved())
    assert snap["axes_not_selectable"] == []
    assert backend.verify(_config(fft_backend="xla"), _resolved(), strict=True) == []


# --------------------------------------------------------------- jax_env ----
def test_jax_env_sets_x64_from_the_config_precision():
    assert _config().jax_env()["JAX_ENABLE_X64"] == "1"
    assert _config(precision_override="complex64").jax_env()["JAX_ENABLE_X64"] == "0"


def test_jax_env_pins_xla_threads():
    """OMP_NUM_THREADS does not reach XLA; without this a threads=1 run could
    quietly use every core."""
    flags = _config(threads=1).jax_env()["XLA_FLAGS"]
    assert "--xla_cpu_multi_thread_eigen=false" in flags
    assert "intra_op_parallelism_threads=1" in flags
    assert "intra_op_parallelism_threads=8" in _config(threads=8).jax_env()["XLA_FLAGS"]


def test_full_env_lets_a_config_override_the_defaults():
    cfg = _config(env={"JAX_ENABLE_X64": "0"})
    assert cfg.full_env()["JAX_ENABLE_X64"] == "0"


def test_gpu_configs_do_not_get_cpu_thread_flags():
    assert "XLA_FLAGS" not in _config(device="cuda:0").jax_env()


# ------------------------------------------------------------ latest_axes ----
def test_latest_axes_prefers_the_newest_declaration():
    """A superseded caveat must not survive; it is metadata, not a measurement."""
    from dragrace.report import latest_axes

    rows = [
        {"adapter": "dlux", "config": "cpu_xla_1t", "utc": "2026-08-01T00:00:00+00:00",
         "path": "a", "axes_not_selectable": ["fft", "blas"]},
        {"adapter": "dlux", "config": "cpu_xla_1t", "utc": "2026-08-14T00:00:00+00:00",
         "path": "b", "axes_not_selectable": ["blas", "threads"]},
    ]
    assert latest_axes(rows) == {("dlux", "cpu_xla_1t"): {"blas", "threads"}}


def test_latest_axes_is_keyed_by_config_too():
    """dLux cannot honour fft=numpy, but on an XLA config the two agree."""
    from dragrace.report import latest_axes

    rows = [
        {"adapter": "dlux", "config": "cpu_numpy_1t", "utc": "2026-08-14T00:00:00+00:00",
         "path": "a", "axes_not_selectable": ["fft", "blas", "threads"]},
        {"adapter": "dlux", "config": "cpu_xla_1t", "utc": "2026-08-14T00:00:00+00:00",
         "path": "b", "axes_not_selectable": ["blas", "threads"]},
    ]
    axes = latest_axes(rows)
    assert axes[("dlux", "cpu_numpy_1t")] == {"fft", "blas", "threads"}
    assert axes[("dlux", "cpu_xla_1t")] == {"blas", "threads"}
