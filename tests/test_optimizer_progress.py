from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import optimizer as optimizer_module
from engine.optimizer import ForceFieldOptimizer


class OptimizerProgressTests(unittest.TestCase):
    def test_warm_start_is_counted_as_an_additional_evaluation(self) -> None:
        optimizer = ForceFieldOptimizer.__new__(ForceFieldOptimizer)
        optimizer.param_names = ["N_charge"]
        optimizer.param_space = [("N_charge", -0.8, -0.2)]
        optimizer.config = {
            "atom_types": [
                {"label": "N", "params": {"charge": {"init": -0.5}}}
            ],
            "optimization": {},
        }

        seed, source = optimizer._resolve_warm_start()

        self.assertEqual(seed, {"N_charge": -0.5})
        self.assertEqual(source, "atom_types init values")

    def test_explicit_saasbo_never_silently_changes_method(self) -> None:
        with mock.patch.object(optimizer_module, "_SAASBO_AVAILABLE", False):
            with self.assertRaisesRegex(ValueError, "explicitly requested"):
                ForceFieldOptimizer._select_bo_method(20, {"method": "saasbo"})
            self.assertEqual(
                ForceFieldOptimizer._select_bo_method(
                    20, {"method": "auto", "accuracy_priority": False}
                ),
                "turbo",
            )

    def test_early_stop_counter_is_rebuilt_from_checkpoint_history(self) -> None:
        optimizer = ForceFieldOptimizer.__new__(ForceFieldOptimizer)
        optimizer.min_improvement = 0.001
        optimizer.best_history = [0.0200, 0.0150, 0.01499, 0.01499, 0.01498]

        self.assertEqual(optimizer._infer_no_improve_count(), 3)

    def test_best_property_report_prints_and_persists_active_targets(self) -> None:
        optimizer = ForceFieldOptimizer.__new__(ForceFieldOptimizer)
        optimizer.progress_enabled = True
        optimizer.progress_save_csv = True
        optimizer.current_round = 7
        optimizer._active_targets = {
            "a": {"value": 4.2422, "weight": 1.0},
            "density": {"value": 1.3285, "weight": 2.0},
        }
        optimizer.all_results = [
            {
                "success": True,
                "objective": 0.08,
                "round": "round_6",
                "label": "candidate_1",
                "calc_a": 4.30,
                "error_a": 1.36,
                "calc_density": 1.30,
                "error_density": 2.15,
            },
            {
                "success": True,
                "objective": 0.05,
                "round": "round_7",
                "label": "candidate_2",
                "calc_a": 4.25,
                "error_a": 0.18,
                "calc_density": 1.32,
                "error_density": 0.64,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            optimizer.work_dir = directory
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                optimizer._report_best_progress(
                    "round_7", round_time=12.5, improved=True
                )

            output = stdout.getvalue()
            self.assertIn("[IMPROVED]", output)
            self.assertIn("objective=0.050000", output)
            self.assertIn("density", output)

            latest = Path(directory, "progress_latest.csv")
            history = Path(directory, "progress_history.csv")
            self.assertTrue(latest.exists())
            self.assertTrue(history.exists())

            with latest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["property"] for row in rows], ["a", "density"])
            self.assertEqual(float(rows[0]["objective"]), 0.05)

            with history.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stage"], "round_7")
            self.assertEqual(float(rows[0]["calc_density"]), 1.32)


if __name__ == "__main__":
    unittest.main()
