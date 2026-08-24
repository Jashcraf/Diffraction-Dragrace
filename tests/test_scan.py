"""Scan cases: expansion, per-point aggregation, and the plotting series.

The measurement itself is not tested here -- that needs a propagator. What is
tested is everything that decides *what gets measured* and *what a reader sees*,
because those are the parts that fail silently: a scan that quietly drops its
padding ratio, a five-point curve that collapses into one row, a repeated run
that draws a zigzag.
"""
import json
from pathlib import Path

import pytest

from dragrace.case import Case
from dragrace.plots import scan_series, style_for
from dragrace.report import aggregate, best_points, scan_rows

SCAN_CASE = Path("cases/pupil_to_focus/mft_array_scan.yaml")


def _base_dict(**over):
    d = {
        "id": "s", "kind": "pupil_to_focus", "algorithm_class": "matrix_dft",
        "wavelength_m": 500e-9,
        "pupil": {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 64},
        "output": {"focal_length_m": 1.0, "samples_per_lambda_f_d": 4.0,
                   "extent_lambda_f_d": 8.0},
        "accuracy": {"reference": "internal_mft", "max_rel_l2": 1e-10},
    }
    d.update(over)
    return d


# ------------------------------------------------------------- expansion --
def test_shipped_scan_case_expands():
    case = Case.from_yaml(SCAN_CASE)
    subs = case.scan_cases()
    assert case.is_scan and len(subs) == len(case.scan.values)
    assert [s.n_pupil for s in subs] == sorted(case.scan.values)
    assert len({s.id for s in subs}) == len(subs)
    # The focal grid is what makes this a *pupil* array-size scan.
    assert {s.n_focus for s in subs} == {case.n_focus}


def test_scan_points_are_ascending():
    """Cheap points first, so a scan whose largest size dies still has a curve."""
    case = Case.from_dict(_base_dict(scan={"parameter": "n_pupil", "values": [512, 64, 128]}))
    assert [s.n_pupil for s in case.scan_cases()] == [64, 128, 512]


def test_n_pupil_scan_preserves_the_padding_ratio():
    """A case declaring 2x padding keeps it at every size.

    The scan value IS n_pupil -- the array size, matching the parameter's name
    and the plotted axis -- and the aperture is derived from it. A free-space
    case with a 4x guard band therefore keeps its guard band as N grows instead
    of watching it shrink to nothing.
    """
    d = _base_dict(algorithm_class="fft",
                   scan={"parameter": "n_pupil", "values": [128, 256]})
    d["pupil"] = {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 128}
    subs = Case.from_dict(d).scan_cases()
    assert [(s.n_across, s.n_pupil) for s in subs] == [(64, 128), (128, 256)]


def test_scan_values_must_respect_the_padding_ratio():
    """A value that cannot be divided by the padding ratio has no aperture."""
    d = _base_dict(scan={"parameter": "n_pupil", "values": [64, 66]})
    d["pupil"] = {"diameter_m": 0.01, "samples_across_diameter": 16, "array_samples": 64}
    with pytest.raises(ValueError, match="divisible by the padding ratio"):
        Case.from_dict(d)


def test_n_focus_scan_moves_the_focal_grid_only():
    case = Case.from_dict(_base_dict(scan={"parameter": "n_focus", "values": [64, 256]}))
    subs = case.scan_cases()
    assert [s.n_focus for s in subs] == [64, 256]
    assert {s.n_pupil for s in subs} == {64}


def test_non_scan_case_is_its_own_single_point():
    case = Case.from_yaml("cases/pupil_to_focus/mft_n1024_q4.yaml")
    assert not case.is_scan
    assert case.scan_cases() == [case]
    assert case.total_timeout_s == case.execution.timeout_s


def test_scan_timeout_is_per_point():
    """The runner budgets timeout_s per measurement; adding sizes must not
    silently start killing the run."""
    case = Case.from_dict(_base_dict(scan={"parameter": "n_pupil", "values": [64, 128, 256]}))
    assert case.total_timeout_s == 3 * case.execution.timeout_s


@pytest.mark.parametrize("scan, match", [
    ({"parameter": "n_samples", "values": [64]}, "scan.parameter"),
    ({"parameter": "n_pupil", "values": []}, "empty"),
    ({"parameter": "n_pupil", "values": [64, 64]}, "duplicates"),
    ({"parameter": "n_pupil", "values": [65]}, "even"),
    ({"parameter": "n_pupil", "values": [4]}, ">= 8"),
])
def test_malformed_scans_fail_at_load(scan, match):
    with pytest.raises(ValueError, match=match):
        Case.from_dict(_base_dict(scan=scan))


def test_non_integer_padding_ratio_is_rejected():
    d = _base_dict(scan={"parameter": "n_pupil", "values": [64, 128]})
    d["pupil"] = {"diameter_m": 0.01, "samples_across_diameter": 64, "array_samples": 100}
    with pytest.raises(ValueError, match="padding ratio"):
        Case.from_dict(d)


# ------------------------------------------------------------ aggregation --
def _result(adapter, medians, status="ok", case="mft_array_scan", run="r1"):
    """A result.json shaped like the worker writes one for a scan."""
    return {
        "schema_version": 1, "case_id": case, "config_id": "cpu_numpy_1t",
        "mode": "timing", "status": status,
        "adapter": {"name": adapter},
        "machine": {"id": "sha256:aaaa", "cpu": "test"},
        "scan": {
            "parameter": "n_pupil", "values": sorted(medians),
            "points": [
                {"scan_value": n, "case_id": f"{case}@n_pupil={n}", "n_pupil": n,
                 "n_focus": 128, "status": "ok",
                 "setup": {"build_s": 0.01, "first_call_s": 0.02},
                 "flops": {"ideal": {"flops": 8.0 * n * n * 128}},
                 "accuracy": {"rel_l2": 0.0, "gate": "pass"},
                 "timing": {"traced": False, "device_compute_stats": {
                     "median": t, "min": t * 0.98, "p95": t * 1.05, "iqr": t * 0.01}}}
                for n, t in sorted(medians.items())
            ],
        },
    }


def _write(tmp_path, res, run="r1"):
    d = (tmp_path / "raw" / "sha256_aaaa" / run / res["adapter"]["name"]
         / res["config_id"] / res["case_id"] / res["mode"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res))


def test_aggregate_emits_one_row_per_scan_point(tmp_path):
    _write(tmp_path, _result("numpy_baseline", {128: 0.001, 256: 0.004, 512: 0.016}))
    rows = aggregate(tmp_path)
    assert len(rows) == 3
    assert [r["scan_value"] for r in rows] == [128, 256, 512]
    assert {r["scan_param"] for r in rows} == {"n_pupil"}
    assert all(r["case"] == "mft_array_scan" for r in rows)
    assert rows[1]["median_s"] == 0.004 and rows[1]["p95_s"] == pytest.approx(0.0042)


def test_plain_result_still_yields_exactly_one_row(tmp_path):
    """The scan path must not change what a non-scan result aggregates to."""
    res = {"schema_version": 1, "case_id": "mft_n1024_q4", "config_id": "cpu_numpy_1t",
           "mode": "timing", "status": "ok", "adapter": {"name": "prysm"},
           "machine": {"id": "sha256:aaaa", "cpu": "test"},
           "timing": {"traced": False, "device_compute_stats": {"median": 0.03, "min": 0.029}}}
    _write(tmp_path, res)
    rows = aggregate(tmp_path)
    assert len(rows) == 1
    assert rows[0]["scan_value"] is None and rows[0]["median_s"] == 0.03


def test_failed_point_keeps_its_own_status(tmp_path):
    res = _result("poppy", {128: 0.001, 256: 0.004})
    res["status"] = "partial"
    res["scan"]["points"][1].update(status="accuracy_fail", reason="gate", timing={})
    _write(tmp_path, res)
    rows = sorted(aggregate(tmp_path), key=lambda r: r["scan_value"])
    assert [r["status"] for r in rows] == ["ok", "accuracy_fail"]
    assert len(scan_rows(rows)) == 1, "a failed point is never plottable"


def test_repeated_runs_collapse_to_the_fastest(tmp_path):
    """Two run_ids of the same point must not draw two points at one x."""
    _write(tmp_path, _result("prysm", {128: 0.002, 256: 0.008}), run="r1")
    _write(tmp_path, _result("prysm", {128: 0.001, 256: 0.009}), run="r2")
    rows = aggregate(tmp_path)
    assert len(rows) == 4
    best = sorted(best_points(rows), key=lambda r: r["scan_value"])
    assert [r["median_s"] for r in best] == [0.001, 0.008]


# ---------------------------------------------------------------- plotting --
def test_scan_series_groups_by_machine_then_adapter(tmp_path):
    _write(tmp_path, _result("prysm", {128: 0.002, 256: 0.008}))
    _write(tmp_path, _result("lentil", {128: 0.003, 256: 0.009}))
    groups = scan_series(aggregate(tmp_path))
    assert len(groups) == 1, "one figure per (machine, contract, case, config, mode, param)"
    (key,) = groups
    assert key[0] == "sha256:aaaa" and key[6] == "n_pupil"
    lines = groups[key]
    assert set(lines) == {"prysm", "lentil"}
    assert [r["scan_value"] for r in lines["prysm"]] == [128, 256]


def test_adapter_colours_are_stable_and_distinct():
    """Colour follows the adapter, so filtering one out never repaints another."""
    names = ["numpy_baseline", "prysm", "hcipy", "poppy", "lentil", "proper",
             "dlux", "abcdlux"]
    colours = [style_for(n)[0] for n in names]
    assert len(set(colours)) == len(names)
    assert style_for("prysm") == style_for("prysm")


def test_contracts_are_never_merged(tmp_path):
    """A primitive-v1 point and an idiomatic-v1 point at the same size are two
    different measurements, and must not collapse into one curve."""
    old = _result("poppy", {128: 0.0013, 256: 0.0035})
    new = _result("poppy", {128: 0.0033, 256: 0.0067})
    new["measurement_contract"] = "idiomatic-v1"
    _write(tmp_path, old, run="r1")
    _write(tmp_path, new, run="r2")

    rows = aggregate(tmp_path)
    assert {r["contract"] for r in rows} == {"primitive-v1", "idiomatic-v1"}
    # best_points keeps the fastest *within* a contract, never across them.
    best = best_points(rows)
    assert len(best) == 4
    assert sorted(r["median_s"] for r in best) == [0.0013, 0.0033, 0.0035, 0.0067]
    assert len(scan_series(rows)) == 2, "one figure per contract"


def test_missing_contract_marker_reads_as_primitive(tmp_path):
    """Results written before contracts existed measured the transform."""
    _write(tmp_path, _result("prysm", {128: 0.002}))
    (row,) = aggregate(tmp_path)
    assert row["contract"] == "primitive-v1"


# ---------------------------------------------------- the n_zernike axis --
# The one scan axis that moves neither grid. It shares the machinery with
# n_pupil and n_focus and almost none of their constraints, which is exactly
# where a shared code path goes wrong quietly.
NZERNIKE_CASE = Path("cases/phase_retrieval/pr_nzernike_n256_numeric_scan.yaml")


def test_n_zernike_scan_moves_only_the_parameter_count():
    """Every grid is identical at every point; only P moves. If a pupil or a
    focal grid drifted along this axis the curve would be measuring two things
    at once and would still look perfectly reasonable."""
    case = Case.from_yaml(NZERNIKE_CASE)
    subs = case.scan_cases()

    assert [s.n_zernike for s in subs] == sorted(case.scan.values)
    assert {s.n_pupil for s in subs} == {256}
    assert {s.n_across for s in subs} == {256}
    assert {s.n_focus for s in subs} == {64}
    assert {s.retrieval.gradient for s in subs} == {"numerical"}


def test_n_zernike_accepts_counts_a_grid_axis_would_reject():
    """3 and 15 are fine parameter counts and impossible array sizes. The even
    and >= 8 rules belong to a grid's centring convention, and applying them
    here would reject most of the shipped scan."""
    case = Case.from_yaml(NZERNIKE_CASE)
    assert 3 in case.scan.values and 15 in case.scan.values
    assert any(v % 2 for v in case.scan.values), "the shipped scan has odd values"


def test_n_zernike_scan_rejected_for_a_non_retrieval_case():
    d = _base_dict(scan={"parameter": "n_zernike", "values": [3, 6]})
    with pytest.raises(ValueError, match="n_zernike"):
        Case.from_dict(d)


def test_n_zernike_scan_rejects_a_zero_parameter_count():
    import yaml
    d = yaml.safe_load(Path(NZERNIKE_CASE).read_text())
    d["scan"]["values"] = [0, 3]
    with pytest.raises(ValueError, match=">= 1"):
        Case.from_dict(d)
