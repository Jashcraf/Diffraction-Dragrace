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


def test_jax_env_does_not_pretend_to_pin_xla_threads():
    """This test used to assert the opposite, and the assertion was false.

    It required XLA_FLAGS to carry
    `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=N`, and
    neither half did anything. Measured on jaxlib 0.10.2 at N_p=1024, dLux ran
    at cpu/wall = 10.06 with no XLA_FLAGS, 10.04 with the eigen flag and 9.92
    with the full string -- ~10 cores throughout, on a config labelled
    threads=1. Worse, `intra_op_parallelism_threads` is not an XLA flag at all;
    it survived only because it lacked the `--` prefix and was discarded as a
    positional, and adding the prefix aborts the process outright.

    So the config must NOT emit it, and the threads axis is enforced by CPU
    affinity in the worker instead.
    """
    assert "XLA_FLAGS" not in _config(threads=1).jax_env()
    assert "XLA_FLAGS" not in _config(threads=8).jax_env()
    # x64 is still the config's business and still has to be an env var.
    assert _config(threads=1).jax_env()["JAX_ENABLE_X64"] == "1"


def test_worker_pins_cpu_affinity_to_the_requested_thread_count():
    """The mechanism that actually enforces `threads`, since no library-level
    knob reaches XLA. Affinity is a property of the process, so nothing can opt
    out of it."""
    import os

    from dragrace.worker import _pin_cpus

    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("no sched_setaffinity on this platform")

    original = os.sched_getaffinity(0)
    try:
        if len(original) < 2:
            pytest.skip("need at least 2 cores to observe a restriction")
        _pin_cpus(_config(threads=1))
        assert len(os.sched_getaffinity(0)) == 1
    finally:
        os.sched_setaffinity(0, original)


def test_gpu_configs_are_not_pinned():
    """The device does the work; throttling the host thread would only slow
    dispatch."""
    import os

    from dragrace.worker import _pin_cpus

    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("no sched_setaffinity on this platform")
    original = os.sched_getaffinity(0)
    try:
        _pin_cpus(_config(threads=1, device="cuda:0"))
        assert os.sched_getaffinity(0) == original
    finally:
        os.sched_setaffinity(0, original)


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


# ------------------------------------------- one config, two environments ----
def test_gpu_config_routes_dlux_to_the_jax_environment():
    """gpu_f64 is served by two envs because CuPy and pip's CUDA JAX cannot
    share an interpreter. Splitting the config id instead would put
    dLux-on-CUDA and prysm-on-CUDA on different boards."""
    from pathlib import Path

    cfg = Config.from_yaml(Path(__file__).parent.parent / "configs" / "gpu_f64.yaml")
    assert cfg.env_for("dlux") == "dragrace-gpu-jax"
    assert cfg.env_for("prysm") == "dragrace-gpu-cupy"
    assert cfg.env_for("poppy") == "dragrace-gpu-cupy"
    assert cfg.env_for(None) == "dragrace-gpu-cupy"


def test_cpu_configs_serve_every_adapter_from_one_environment():
    cfg = _config()
    assert cfg.conda_env_by_adapter == {}
    assert cfg.env_for("dlux") == cfg.env_for("prysm") == cfg.conda_env


def test_cuda_path_points_at_the_conda_target_dir(tmp_path):
    """CuPy appends /include to CUDA_PATH, and conda-forge puts the headers
    under $PREFIX/targets/<arch>/ -- so the prefix itself is the wrong answer
    and produces 'Failed to find CUDA headers' at the first kernel launch."""
    from dragrace.runner import cuda_path_for

    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    target = prefix / "targets" / "x86_64-linux"
    (target / "include").mkdir(parents=True)
    (target / "include" / "cuda_runtime.h").write_text("")
    assert cuda_path_for(str(prefix / "bin" / "python")) == str(target)


def test_cuda_path_is_none_without_conda_cuda(tmp_path):
    """A CPU environment must not have CUDA_PATH rewritten under it."""
    from dragrace.runner import cuda_path_for

    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    assert cuda_path_for(str(prefix / "bin" / "python")) is None


# --------------------------------------------- per-adapter retrieval device --
def test_retrieval_gpu_refusal_is_per_adapter():
    """The board was CPU-only when no retrieval chain was device-aware. Now
    that three are, the refusal has to name the adapter rather than the board,
    or a code with a real GPU path is silently excluded from it."""
    from pathlib import Path

    from dragrace import adapter as adapter_mod
    from dragrace.case import Case

    repo = Path(__file__).parent.parent
    adapter_mod.discover(repo / "adapters")
    case = Case.from_yaml(
        repo / "cases" / "phase_retrieval" / "pr_zernike11_analytic_scan.yaml"
    ).scan_cases()[0]
    gpu = Config.from_yaml(repo / "configs" / "gpu_f64.yaml")

    for name in ("prysm", "dlux"):
        assert adapter_mod.get(name).retrieval_support(case, gpu) is True
    for name in ("hcipy", "lentil", "proper"):
        sup = adapter_mod.get(name).retrieval_support(case, gpu)
        assert not sup and "gradient" in sup.reason or "GPU" in sup.reason
