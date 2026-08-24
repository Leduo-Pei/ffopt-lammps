from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from engine import optimizer as optimizer_module
from engine.optimizer import ForceFieldOptimizer


def _config() -> dict:
    return {
        "manifest": {"system_name": "coverage_test"},
        "targets": {"a": {"value": 2.0, "tolerance": 0.1, "weight": 1.0}},
        "lammps": {"compute_surface": False},
        "optimization": {"accuracy_priority": True},
    }


class _Runner:
    parameter_difference_constraints = []

    def __init__(self, results):
        self.results = results

    def evaluate_batch(self, _parameters, _directory):
        return self.results


def _result(*, p: float, a: float, fit_objective: float):
    return SimpleNamespace(
        success=True,
        objective=fit_objective,
        error_msg="",
        params={"p": p},
        properties={"a": a},
        per_property_error={"a": 100.0 * abs(a - 2.0) / 2.0},
        obj_structural=fit_objective,
        obj_surface=0.0,
    )


def _bare_optimizer(tmp_path: Path) -> ForceFieldOptimizer:
    optimizer = object.__new__(ForceFieldOptimizer)
    optimizer.coverage_enabled = True
    optimizer.coverage_cfg = {"archive_target": 3}
    optimizer.coverage_surrogate_fallbacks = []
    optimizer.config = _config()
    optimizer.param_names = ["p"]
    optimizer.param_space = [("p", 0.0, 1.0)]
    optimizer.n_params = 1
    optimizer.bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizer.current_round = 1
    optimizer.work_dir = str(tmp_path)
    optimizer.all_results = []
    optimizer.all_feasible = []
    optimizer.all_X = torch.empty(0, 1, dtype=torch.double)
    optimizer.train_X = torch.empty(0, 1, dtype=torch.double)
    optimizer.train_Y = torch.empty(0, 1, dtype=torch.double)
    optimizer.train_Y2 = torch.empty(0, 2, dtype=torch.double)
    optimizer._best_valid_obj = float("inf")
    optimizer._pending_selection_roles = {
        (0.2,): "feasible_interior",
        (0.8,): "constraint_boundary",
    }
    optimizer._pending_selection_guidance = {
        (0.2,): "model_guided",
        (0.8,): "degraded:objective_gp",
    }
    optimizer.penalty_cap_multiplier = 3.0
    optimizer.pareto_active = False
    optimizer.pareto_posthoc = False
    optimizer.bo_method = "gp"
    optimizer.all_feasible = []
    optimizer._update_feasibility_classifier = lambda: None
    return optimizer


def test_evaluation_records_continuous_gates_and_exact_classifier_labels(tmp_path):
    optimizer = _bare_optimizer(tmp_path)
    optimizer.runner = _Runner([
        _result(p=0.2, a=2.05, fit_objective=0.20),
        _result(p=0.8, a=2.20, fit_objective=0.05),
    ])

    optimizer._evaluate_and_record([{"p": 0.2}, {"p": 0.8}], "round_1")

    assert optimizer.all_feasible == [True, False]
    assert optimizer.train_Y.squeeze(-1).tolist() == pytest.approx([0.0, 1.0])
    assert optimizer.all_results[0]["structural_ratio_a"] == pytest.approx(0.5)
    assert optimizer.all_results[1]["structural_ratio_a"] == pytest.approx(2.0)
    assert optimizer.all_results[0]["structural_margin_a"] == pytest.approx(0.5)
    assert optimizer.all_results[0]["selection_role"] == "feasible_interior"
    assert optimizer.all_results[1]["selection_role"] == "constraint_boundary"
    assert optimizer.all_results[0]["selection_guidance"] == "model_guided"
    assert optimizer.all_results[1]["selection_guidance"] == (
        "degraded:objective_gp"
    )


def test_summary_outputs_choose_feasible_fit_representative_and_report_diversity(
    tmp_path,
):
    optimizer = _bare_optimizer(tmp_path)
    optimizer.all_results = [
        {
            "success": True,
            "objective": 0.01,
            "fit_objective": 0.01,
            "structural_feasible": False,
            "structural_constraint_violation": 0.10,
            "structural_band_max_ratio": 1.10,
            "selection_role": "constraint_boundary",
            "p": 0.1,
            "calc_a": 2.11,
            "error_a": 5.5,
        },
        {
            "success": True,
            "objective": 0.20,
            "fit_objective": 0.20,
            "structural_feasible": True,
            "structural_constraint_violation": 0.0,
            "structural_band_max_ratio": 0.8,
            "selection_role": "feasible_interior",
            "p": 0.4,
            "calc_a": 2.08,
            "error_a": 4.0,
        },
        {
            "success": True,
            "objective": 0.10,
            "fit_objective": 0.10,
            "structural_feasible": True,
            "structural_constraint_violation": 0.0,
            "structural_band_max_ratio": 0.2,
            "selection_role": "global_novelty",
            "p": 0.9,
            "calc_a": 2.02,
            "error_a": 1.0,
        },
    ]
    optimizer.all_feasible = [False, True, True]
    optimizer.current_round = 2
    best = optimizer._select_report_best(optimizer.all_results)

    optimizer._save_summary_files(best, 1.0, optimizer.all_results, [])

    expected = {
        "all_results.csv",
        "feasible_archive.csv",
        "coverage_anchors.csv",
        "coverage_summary.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    summary = json.loads((tmp_path / "coverage_summary.json").read_text())
    assert summary["representative"]["structural_feasible"] is True
    assert summary["schema"] == "ffopt-structural-coverage-summary-v2"
    assert summary["surrogate_guidance"] == {
        "status": "healthy",
        "fallback_event_count": 0,
        "fallback_round_count": 0,
        "fallback_rounds": [],
        "fallback_components": [],
        "fallback_events": [],
    }
    assert summary["representative"]["parameters"] == {"p": 0.9}
    assert summary["counts"]["structurally_feasible"] == 2
    assert summary["coverage_settings"] == {
        "archive_target": 3,
        "candidate_pool": 16384,
        "feasible_fraction": 0.5,
        "boundary_fraction": 0.25,
        "uncertainty_fraction": 0.15,
        "global_fraction": 0.1,
        "minimum_novelty": 1.0e-6,
        "anchor_max_band_ratio": 3.0,
        "minimum_archive": 3,
        "minimum_boundary_anchors": 3,
        "maximum_fallback_rounds": 0,
        "minimum_archive_separation_normalized": 0.001,
        "minimum_weak_span_normalized": 0.001,
        "random_seed": 42,
    }
    assert summary["archive_diversity"][
        "minimum_pair_distance_normalized"
    ] == pytest.approx(0.5)
    archive = pd.read_csv(tmp_path / "feasible_archive.csv")
    assert archive.iloc[0]["p"] == pytest.approx(0.9)
    assert archive.iloc[0]["archive_selection_role"] == (
        "best_fit_structural_feasible"
    )


def test_capped_coverage_gp_training_data_owns_mutable_numeric_buffer(tmp_path):
    optimizer = _bare_optimizer(tmp_path)
    total = 513
    optimizer.train_X = torch.linspace(
        0.0, 1.0, total, dtype=torch.double
    ).reshape(-1, 1)
    optimizer.train_Y = torch.linspace(
        0.0, 1.0, total, dtype=torch.double
    ).reshape(-1, 1)
    optimizer.all_feasible = [index % 3 == 0 for index in range(total)]
    optimizer.all_results = [
        {
            "success": True,
            "fit_objective": None if index % 17 == 0 else index / total,
        }
        for index in range(total)
    ]

    selected_x, selected_y = optimizer._get_coverage_gp_training_data(32)

    assert selected_x.shape == (32, 1)
    assert selected_y.shape == (32, 1)
    assert torch.isfinite(selected_x).all()
    assert torch.isfinite(selected_y).all()


def test_coverage_surrogate_fallback_is_persisted_in_summary(tmp_path):
    optimizer = _bare_optimizer(tmp_path)
    optimizer._record_coverage_surrogate_fallback(
        "objective_gp", ValueError("synthetic failure")
    )
    optimizer._record_coverage_surrogate_fallback(
        "feasibility_classifier",
        RuntimeError("synthetic classifier failure"),
        fallback="objective_gp_mean",
    )
    optimizer.all_results = [{
        "success": True,
        "objective": 0.1,
        "fit_objective": 0.1,
        "structural_feasible": True,
        "structural_constraint_violation": 0.0,
        "structural_band_max_ratio": 0.5,
        "selection_role": "global_novelty",
        "p": 0.5,
        "calc_a": 2.0,
        "error_a": 0.0,
    }]
    optimizer.all_feasible = [True]
    best = optimizer._select_report_best(optimizer.all_results)

    optimizer._save_summary_files(best, 1.0, optimizer.all_results, [])

    summary = json.loads((tmp_path / "coverage_summary.json").read_text())
    assert summary["surrogate_guidance"]["status"] == "failed"
    assert summary["surrogate_guidance"]["fallback_event_count"] == 2
    assert summary["surrogate_guidance"]["fallback_round_count"] == 1
    assert summary["surrogate_guidance"]["fallback_rounds"] == [1]
    assert summary["surrogate_guidance"]["fallback_components"] == [
        "feasibility_classifier",
        "objective_gp",
    ]
    assert summary["surrogate_guidance"]["fallback_events"] == [
        {
            "round": 1,
            "component": "objective_gp",
            "error_type": "ValueError",
            "message": "synthetic failure",
            "fallback": "novelty",
        },
        {
            "round": 1,
            "component": "feasibility_classifier",
            "error_type": "RuntimeError",
            "message": "synthetic classifier failure",
            "fallback": "objective_gp_mean",
        },
    ]


def _write_three_dimensional_coverage_summary(
    tmp_path: Path, *, strict_cloud_width: float
) -> dict:
    optimizer = _bare_optimizer(tmp_path)
    optimizer.param_names = ["p1", "p2", "p3"]
    optimizer.param_space = [
        (name, 0.0, 1.0) for name in optimizer.param_names
    ]
    optimizer.n_params = 3
    optimizer.bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double
    )
    optimizer.coverage_cfg = {
        "archive_target": 12,
        "minimum_archive": 8,
        "minimum_boundary_anchors": 4,
    }
    strict_rows = []
    for index, corner in enumerate(product((0.0, 1.0), repeat=3)):
        values = 0.5 + strict_cloud_width * (np.asarray(corner) - 0.5)
        strict_rows.append({
            "success": True,
            "objective": 0.01 + index * 0.001,
            "fit_objective": 0.01 + index * 0.001,
            "structural_feasible": True,
            "structural_constraint_violation": 0.0,
            "structural_band_max_ratio": 0.5,
            "selection_role": "feasible_interior",
            "selection_guidance": "model_guided",
            "p1": values[0],
            "p2": values[1],
            "p3": values[2],
            "calc_a": 2.0,
            "error_a": 0.0,
        })
    outside_points = [
        (0.05, 0.5, 0.5),
        (0.95, 0.5, 0.5),
        (0.5, 0.05, 0.5),
        (0.5, 0.5, 0.95),
    ]
    outside_rows = [
        {
            "success": True,
            "objective": 0.2 + index * 0.01,
            "fit_objective": 0.2 + index * 0.01,
            "structural_feasible": False,
            "structural_constraint_violation": 0.2,
            "structural_band_max_ratio": 1.2,
            "selection_role": "constraint_boundary",
            "selection_guidance": "model_guided",
            "p1": values[0],
            "p2": values[1],
            "p3": values[2],
            "calc_a": 2.2,
            "error_a": 10.0,
        }
        for index, values in enumerate(outside_points)
    ]
    optimizer.all_results = [*strict_rows, *outside_rows]
    optimizer.all_feasible = [True] * 8 + [False] * 4
    best = optimizer._select_report_best(optimizer.all_results)
    optimizer._save_summary_files(best, 1.0, optimizer.all_results, [])
    return json.loads(
        (tmp_path / "coverage_summary.json").read_text(encoding="utf-8")
    )


def test_diverse_full_rank_archive_is_canonical_coverage(tmp_path):
    summary = _write_three_dimensional_coverage_summary(
        tmp_path, strict_cloud_width=0.8
    )

    assert summary["coverage_quality"]["status"] == "canonical_usable"
    assert summary["coverage_evidence"]["violations"] == []
    observed = summary["coverage_evidence"]["observed"]
    assert observed["affine_rank"] == 3
    assert observed["minimum_pair_distance_normalized"] > 0.001
    assert observed["weak_direction_rms_span_normalized"] > 0.001


def test_nearly_coincident_full_rank_archive_is_recovery_only(tmp_path):
    summary = _write_three_dimensional_coverage_summary(
        tmp_path, strict_cloud_width=1.0e-5
    )

    assert summary["coverage_evidence"]["observed"]["affine_rank"] == 3
    assert summary["coverage_quality"]["status"] == "recovery_required"
    assert set(summary["coverage_evidence"]["violations"]) >= {
        "feasible_archive_separation_below_minimum",
        "feasible_archive_weak_span_below_minimum",
    }


class _PosteriorGP:
    likelihood = object()

    def __init__(self, *_args, **_kwargs):
        pass

    def posterior(self, pool):
        count = len(pool)
        return SimpleNamespace(
            mean=torch.arange(count, dtype=torch.double).reshape(-1, 1),
            variance=torch.full((count, 1), 0.25, dtype=torch.double),
        )


def _proposal_optimizer(tmp_path: Path) -> ForceFieldOptimizer:
    optimizer = _bare_optimizer(tmp_path)
    optimizer.coverage_cfg = {"candidate_pool": 4}
    optimizer.batch_size = 2
    optimizer.train_X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double)
    optimizer.train_Y = torch.tensor([[0.4], [0.0], [0.8]], dtype=torch.double)
    optimizer.all_X = optimizer.train_X.clone()
    optimizer.current_round = 7
    optimizer._generate_lhs = lambda count, seed_offset=0: [
        {"p": index / max(count - 1, 1)} for index in range(count)
    ]
    optimizer._get_gp_training_data = lambda: (
        optimizer.train_X, optimizer.train_Y
    )
    return optimizer


def _capture_coverage_selection(monkeypatch):
    captured = {}

    def select(pool, **kwargs):
        captured.update(kwargs)
        return pool[:2], ["feasible_interior", "model_uncertainty"]

    monkeypatch.setattr(optimizer_module, "select_coverage_batch", select)
    monkeypatch.setattr(optimizer_module, "SingleTaskGP", _PosteriorGP)
    monkeypatch.setattr(
        optimizer_module, "ExactMarginalLogLikelihood", lambda *_args: object()
    )
    monkeypatch.setattr(optimizer_module, "fit_gpytorch_mll", lambda *_args: None)
    return captured


def test_failed_classifier_uses_objective_gp_probability(tmp_path, monkeypatch):
    optimizer = _proposal_optimizer(tmp_path)

    class BrokenClassifier:
        def predict_proba(self, _pool):
            raise RuntimeError("classifier unavailable")

    optimizer._feasibility_model = BrokenClassifier()
    captured = _capture_coverage_selection(monkeypatch)

    optimizer._propose_batch_coverage()

    means = np.arange(4, dtype=float)
    scale = np.median(np.abs(means))
    assert captured["feasible_probability"] == pytest.approx(
        np.exp(-np.maximum(means, 0.0) / scale)
    )
    assert captured["model_uncertainty"] == pytest.approx([0.5] * 4)
    assert optimizer.coverage_surrogate_fallbacks == [{
        "round": 7,
        "component": "feasibility_classifier",
        "error_type": "RuntimeError",
        "message": "classifier unavailable",
        "fallback": "objective_gp_mean",
    }]
    assert set(optimizer._pending_selection_guidance.values()) == {
        "degraded:feasibility_classifier"
    }


def test_failed_objective_gp_preserves_classifier_probability(
    tmp_path, monkeypatch
):
    optimizer = _proposal_optimizer(tmp_path)
    classifier_probability = np.array([0.1, 0.3, 0.7, 0.9])

    class WorkingClassifier:
        def predict_proba(self, _pool):
            return np.column_stack((1.0 - classifier_probability, classifier_probability))

    class BrokenGP:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("GP unavailable")

    optimizer._feasibility_model = WorkingClassifier()
    captured = _capture_coverage_selection(monkeypatch)
    monkeypatch.setattr(optimizer_module, "SingleTaskGP", BrokenGP)

    optimizer._propose_batch_coverage()

    assert captured["feasible_probability"] == pytest.approx(
        classifier_probability
    )
    assert captured["model_uncertainty"] == pytest.approx([1.0] * 4)
    assert optimizer.coverage_surrogate_fallbacks == [{
        "round": 7,
        "component": "objective_gp",
        "error_type": "RuntimeError",
        "message": "GP unavailable",
        "fallback": "classifier_probability_with_novelty_uncertainty",
    }]
    assert set(optimizer._pending_selection_guidance.values()) == {
        "degraded:objective_gp"
    }


def test_classifier_training_failure_degrades_next_acquisition(
    tmp_path, monkeypatch
):
    optimizer = _proposal_optimizer(tmp_path)
    del optimizer.__dict__["_update_feasibility_classifier"]
    optimizer.seed = 11
    optimizer.all_X = torch.linspace(
        0.0, 1.0, 10, dtype=torch.double
    ).reshape(-1, 1)
    optimizer.all_feasible = [index % 2 == 0 for index in range(10)]

    class BrokenClassifierPipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            raise RuntimeError("classifier fit failed")

    monkeypatch.setattr(optimizer_module, "Pipeline", BrokenClassifierPipeline)
    optimizer._update_feasibility_classifier()
    assert optimizer._feasibility_model is None
    assert str(optimizer._feasibility_classifier_error) == "classifier fit failed"

    captured = _capture_coverage_selection(monkeypatch)
    optimizer._propose_batch_coverage()

    assert np.ptp(captured["feasible_probability"]) > 0.0
    assert optimizer.coverage_surrogate_fallbacks == [{
        "round": 7,
        "component": "feasibility_classifier",
        "error_type": "RuntimeError",
        "message": "classifier fit failed",
        "fallback": "objective_gp_mean",
    }]
    assert set(optimizer._pending_selection_guidance.values()) == {
        "degraded:feasibility_classifier"
    }


def test_disabled_classifier_uses_objective_gp_without_false_fallback(
    tmp_path, monkeypatch
):
    optimizer = _proposal_optimizer(tmp_path)
    del optimizer.__dict__["_update_feasibility_classifier"]
    optimizer.use_feasibility_classifier = False
    optimizer._feasibility_model = object()
    optimizer._feasibility_classifier_error = RuntimeError("stale failure")

    optimizer._update_feasibility_classifier()
    assert optimizer._feasibility_model is None
    assert optimizer._feasibility_classifier_error is None

    captured = _capture_coverage_selection(monkeypatch)
    optimizer._propose_batch_coverage()

    assert np.ptp(captured["feasible_probability"]) > 0.0
    assert optimizer.coverage_surrogate_fallbacks == []
    assert set(optimizer._pending_selection_guidance.values()) == {
        "model_guided"
    }


def test_noncoverage_report_selection_keeps_legacy_objective_order(tmp_path):
    optimizer = _bare_optimizer(tmp_path)
    optimizer.coverage_enabled = False
    rows = [
        {"objective": 0.4, "structural_feasible": True},
        {"objective": 0.1, "structural_feasible": False},
    ]

    assert optimizer._select_report_best(rows) is rows[1]
