"""The phase-retrieval board: the geometry, the inverse problem, the gradients.

The gradient checks here are the load-bearing ones. A wrong analytic gradient
does not raise, does not fail the accuracy gate on its own, and does not even
stop L-BFGS-B from converging -- it partly absorbs a bad gradient into its step
length. The symptom is a stall, and a stall reads as "this code is slow", which
is exactly the wrong conclusion for a timing board to publish. Both prysm's
hand-written adjoint and dLux's reverse-mode AD are therefore pinned against
central differences on the case they will actually run.
"""
import numpy as np
import pytest
import yaml

from dragrace import adapter as adapter_mod
from dragrace import retrieval as R
from dragrace.case import Case
from dragrace.config import Config
from dragrace.flops import ideal_work
from dragrace.grid import aperture_mask, noll_to_nm

NUMERIC = "cases/phase_retrieval/pr_zernike11_numeric_scan.yaml"
ANALYTIC = "cases/phase_retrieval/pr_zernike11_analytic_scan.yaml"


@pytest.fixture(scope="module")
def case():
    """The smallest scan point of the analytic board -- N=64, cheap enough that
    a 2P-evaluation finite-difference check is affordable in a test."""
    return Case.from_yaml(ANALYTIC).scan_cases()[0]


@pytest.fixture(scope="module")
def config():
    return Config.from_yaml("configs/cpu_numpy_1t.yaml")


# ----------------------------------------------------------------- the case --
def test_the_two_boards_differ_only_in_gradient():
    """The pair exists to isolate one variable. If the physics drifted between
    them, comparing the two figures would silently stop meaning anything."""
    a = yaml.safe_load(open(NUMERIC).read())
    b = yaml.safe_load(open(ANALYTIC).read())
    assert a.pop("id") != b.pop("id")
    assert a.pop("notes") and b.pop("notes")
    assert a["retrieval"].pop("gradient") == "numerical"
    assert b["retrieval"].pop("gradient") == "analytic"
    assert a == b, "the numerical and analytic boards have diverged in physics"


def test_first_eleven_noll_ends_at_primary_spherical():
    """'the first 11, up to primary spherical' -- the two readings must agree,
    and they do only on Noll ordering starting at piston."""
    r = Case.from_yaml(ANALYTIC).retrieval
    assert r.noll_indices == list(range(1, 12))
    assert noll_to_nm(11) == (4, 0), "Noll 11 must be primary spherical"
    assert noll_to_nm(1) == (0, 0), "Noll 1 must be piston"


def test_retrieval_case_rejects_a_static_aberration():
    d = yaml.safe_load(open(ANALYTIC).read())
    d["pupil"]["aberration"] = {"coefficients": {4: 0.1}}
    with pytest.raises(ValueError, match="pupil.aberration"):
        Case.from_dict(d)


def test_retrieval_case_rejects_single_precision():
    d = yaml.safe_load(open(ANALYTIC).read())
    d["dtype"] = "complex64"
    d["accuracy"]["max_rel_l2"] = 1e-3
    with pytest.raises(ValueError, match="complex128"):
        Case.from_dict(d)


def test_obstruction_requires_an_obstructed_aperture():
    d = yaml.safe_load(open(ANALYTIC).read())
    d["pupil"]["aperture"] = "circular"
    with pytest.raises(ValueError, match="obstructed"):
        Case.from_dict(d)


# -------------------------------------------------------------- the geometry --
def test_pupil_has_an_obscuration_and_two_vanes(case):
    amp = aperture_mask(case)
    assert set(np.unique(amp)) <= {0.0, 1.0}, "the injected mask is hard-edged"

    n = case.n_pupil
    ob = case.pupil.obstruction
    assert amp[n // 2, n // 2] == 0.0, "the secondary obscuration is missing"
    # Open somewhere on the annulus, blocked along each vane's own direction.
    for angle in ob.spider_angles_deg:
        a = np.deg2rad(angle)
        r = 0.35                                    # between eps/2=0.15 and 0.5
        i = int(round(n / 2 + r * n * np.sin(a)))
        j = int(round(n / 2 + r * n * np.cos(a)))
        assert amp[i, j] == 0.0, f"no vane found at {angle} deg"
        # ..and open on the opposite side, which is what `radius` means.
        assert amp[n - i, n - j] == 1.0, (
            f"the vane at {angle} deg spans the full diameter; `spider_span: "
            f"radius` is what breaks the twin ambiguity")


def test_pupil_is_not_centro_symmetric(case):
    """The whole reason the retrieval is well posed. phi(x) and -phi(-x) give
    the identical PSF for a centro-symmetric pupil, so a single in-focus image
    cannot separate them."""
    amp = aperture_mask(case)
    # Compare about the centre sample, which is where the symmetry would be.
    core = amp[1:, 1:]
    assert not np.array_equal(core, core[::-1, ::-1])


def test_zernikes_are_unit_rms_over_the_obstructed_pupil(case):
    _, _, _, basis = R.retrieval_parameters(case)
    mask = aperture_mask(case) > 0
    for j, mode in zip(case.retrieval.noll_indices, basis):
        assert np.sqrt(np.mean(mode[mask] ** 2)) == pytest.approx(1.0)
        assert not mode[~mask].any(), f"Noll {j} leaks outside the pupil"


def test_twin_flips_even_radial_orders_and_is_an_involution(case):
    _, theta, _, _ = R.retrieval_parameters(case)
    twin = R.twin_coefficients(case, theta)
    np.testing.assert_allclose(R.twin_coefficients(case, twin), theta)
    for j, a, b in zip(case.retrieval.noll_indices, theta, twin):
        n = noll_to_nm(j)[0]
        assert b == pytest.approx(a if n % 2 else -a), f"Noll {j} (n={n})"


# ------------------------------------------------------------- the reference --
def test_reference_retrieval_recovers_the_truth(case):
    _, theta_true, _, _ = R.retrieval_parameters(case)
    out = R.reference_retrieval(case)
    sl = R.observable_slice(case)
    err = np.linalg.norm(out.theta[sl] - theta_true[sl]) / np.linalg.norm(theta_true[sl])
    assert out.converged
    assert err < case.accuracy.max_rel_l2
    # Measured 8.7e9 at N=64. The floor is not machine epsilon: the pupil's
    # symmetry is broken by a vane a couple of samples wide, so how exactly the
    # truth sits at the bottom of the well is a discretisation detail.
    assert out.loss_final < 1e-8 * out.loss_initial


def test_a_cold_start_would_find_the_twin(case):
    """Pins the measurement behind `initial: truth_perturbed`.

    Breaking the pupil's symmetry makes the truth the unique global minimum but
    leaves the twin only ~1.1e-6 above it, which is invisible from a cold
    start's 1.5e-3. If this ever stops finding the twin, the perturbed start has
    become unnecessary and the case should be simplified -- so it is a test
    rather than a comment.
    """
    from dataclasses import replace

    cold = replace(case, retrieval=replace(case.retrieval, initial="zeros"))
    big = replace(cold, pupil=replace(
        cold.pupil, samples_across_diameter=256, array_samples=256))
    _, theta_true, _, _ = R.retrieval_parameters(big)
    out = R.reference_retrieval(big)
    sl = R.observable_slice(big)
    to_truth = np.linalg.norm(out.theta[sl] - theta_true[sl])
    twin = R.twin_coefficients(big, theta_true)
    to_twin = np.linalg.norm(out.theta[sl] - twin[sl])
    assert to_twin < to_truth, "expected the cold start to land on the twin at N=256"


def test_loss_scale_rejects_a_dark_psf(case):
    with pytest.raises(ValueError, match="non-positive peak"):
        R.loss_scale(np.zeros((4, 4)))


# -------------------------------------------------------------- the gradients --
def _central_differences(fun, theta, h=1e-7):
    eye = np.eye(theta.size)
    return np.array([(fun(theta + h * eye[i])[0] - fun(theta - h * eye[i])[0]) / (2 * h)
                     for i in range(theta.size)])


@pytest.mark.parametrize("name", ["numpy_baseline", "prysm"])
def test_analytic_gradient_matches_central_differences(name, case, config):
    """The check that would have caught the crossed Wirtinger conventions.

    Both chains are correct in their own convention -- numpy_baseline tracks
    dL/dz and uses a plain transpose, prysm's executor.adjoint is the conjugate
    transpose and wants dL/dz* -- and they agree bit-for-bit on the parameter
    gradient. Transcribing an intermediate from one into the other conjugates
    twice, for a gradient wrong by up to 68x per component. It does not raise
    and it still converges, just far more slowly, so on a timing board it would
    have been published as "prysm is slow".

    This is why the boards are defined at the PARAMETER level and never compare
    intermediates -- see docs/gradient_board.md.
    """
    adapter_mod.discover("adapters")
    ad = adapter_mod.get(name)
    assert ad.configure(config)
    state = ad.build(case, config)
    fun, theta = state["fun"], np.asarray(state["theta0"])

    analytic = np.asarray(fun(theta)[1], dtype=float)
    fd = _central_differences(fun, theta)

    sl = R.observable_slice(case)
    # Cosine as well as per-component: a uniform factor (the classic slip) can
    # hide under a loose relative tolerance and shows up immediately here.
    cos = np.dot(analytic[sl], fd[sl]) / (
        np.linalg.norm(analytic[sl]) * np.linalg.norm(fd[sl]))
    assert cos == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(analytic[sl], fd[sl], rtol=2e-6)


@pytest.mark.parametrize("name", ["numpy_baseline", "prysm"])
def test_piston_gradient_is_identically_zero(name, case, config):
    """A PSF cannot see piston, so Noll 1 must come back at exactly zero --
    not merely small. It stays in the parameter vector because 'the first 11'
    means Noll 1..11, and the accuracy gate excludes it rather than pretending
    it was recovered."""
    adapter_mod.discover("adapters")
    ad = adapter_mod.get(name)
    assert ad.configure(config)
    state = ad.build(case, config)
    grad = np.asarray(state["fun"](np.asarray(state["theta0"]))[1], dtype=float)
    assert abs(grad[0]) < 1e-15 * max(np.abs(grad[1:]).max(), 1e-300)


def test_dlux_gradient_matches_central_differences(case, config):
    """Same check as prysm's, against the loss dLux actually minimises.

    `state["loss"]` is the traceable function the compiled loop calls, so this
    differentiates the real thing rather than a reimplementation of it.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("dLux")
    pytest.importorskip("optax")
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX_ENABLE_X64 must be set before the first jax import")

    import jax.numpy as jnp

    adapter_mod.discover("adapters")
    ad = adapter_mod.get("dlux")
    assert ad.configure(config)
    state = ad.build(case, config)
    loss = state["loss"]
    theta = jnp.asarray(state["theta0"])

    analytic = np.asarray(jax.grad(loss)(theta), dtype=float)
    fd = _central_differences(lambda t: (float(loss(jnp.asarray(t))),), np.asarray(theta))

    sl = R.observable_slice(case)
    cos = np.dot(analytic[sl], fd[sl]) / (
        np.linalg.norm(analytic[sl]) * np.linalg.norm(fd[sl]))
    assert cos == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(analytic[sl], fd[sl], rtol=2e-6)
    assert abs(analytic[0]) < 1e-15 * np.abs(analytic[1:]).max(), "piston is unobservable"


def test_dlux_stops_on_the_same_criteria_as_scipy(case, config):
    """optax has no stopping rule of its own, so the adapter transcribes
    scipy's. Without that dLux would run the full iteration cap while the
    scipy-driven codes stopped at ~22, and would look slow for doing four times
    the work."""
    pytest.importorskip("jax")
    pytest.importorskip("dLux")
    pytest.importorskip("optax")

    adapter_mod.discover("adapters")
    ad = adapter_mod.get("dlux")
    assert ad.configure(config)
    out = ad.propagate(ad.build(case, config))
    reference = R.reference_retrieval(case)
    assert out.converged, out.message
    assert out.n_iterations < case.retrieval.max_iterations
    # Not identical -- different line searches -- but the same ballpark, which
    # is what makes the two boards readable against each other.
    assert 0.5 <= out.n_iterations / reference.n_iterations <= 2.0


# ------------------------------------------------------------------- plumbing --
def test_ideal_work_reports_no_total_for_a_retrieval(case):
    """A retrieval case names matrix_dft, and pricing it as one propagation
    would be wrong by the optimiser's evaluation count. flops=0 is what
    suppresses the ideal line on the figure and the ideal row in the table."""
    work = ideal_work(case)
    assert work.flops == 0.0
    assert "measured rather than derivable" in work.detail


def test_adapters_split_cleanly_across_the_two_boards(config):
    adapter_mod.discover("adapters")
    numeric = Case.from_yaml(NUMERIC).scan_cases()[0]
    analytic = Case.from_yaml(ANALYTIC).scan_cases()[0]
    on = {c.retrieval.gradient: {n for n in adapter_mod.available()
                                 if adapter_mod.get(n).supports(c, config)}
          for c in (numeric, analytic)}
    assert {"poppy", "lentil", "proper", "hcipy"} <= on["numerical"]
    assert {"prysm", "dlux"} <= on["analytic"]
    # Nobody appears on both except the harness's own floor.
    assert on["numerical"] & on["analytic"] == {"numpy_baseline"}
