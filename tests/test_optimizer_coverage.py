from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

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


def test_noncoverage_report_selection_keeps_legacy_objective_order(tmp_path):
    optimizer = _bare_optimizer(tmp_path)
    optimizer.coverage_enabled = False
    rows = [
        {"objective": 0.4, "structural_feasible": True},
        {"objective": 0.1, "structural_feasible": False},
    ]

    assert optimizer._select_report_best(rows) is rows[1]
