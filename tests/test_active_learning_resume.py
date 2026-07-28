import json

import pandas as pd

from engine.active_learning import ActiveLearner
from engine.validate_final_parameters import _load_parameters


def test_active_learning_restores_last_completed_round(tmp_path):
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.output_dir = str(tmp_path)
    learner.nn_dir = str(tmp_path / "initial_nn")
    learner.resume = True
    learner.n_rounds = 3
    learner._history = []

    data_dir = tmp_path / "data_round_1"
    model_dir = tmp_path / "nn_round_1"
    data_dir.mkdir()
    model_dir.mkdir()
    pd.DataFrame({"objective": [0.2, 0.1]}).to_csv(
        data_dir / "all_results.csv", index=False
    )
    (model_dir / "forward_nn.pt").write_bytes(b"model")
    (tmp_path / "active_learning_history.json").write_text(
        json.dumps({
            "rounds": [{"round": 1, "n_new_evals": 2}],
            "early_stopped": False,
            "final_round": 1,
        }),
        encoding="utf-8",
    )

    data, start_round, restored_model, early_stopped = learner._restore_progress(
        pd.DataFrame({"objective": [0.3]})
    )
    assert len(data) == 2
    assert start_round == 2
    assert restored_model == str(model_dir)
    assert not early_stopped


def test_active_learning_completed_history_runs_no_more_rounds(tmp_path):
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.output_dir = str(tmp_path)
    learner.nn_dir = str(tmp_path / "initial_nn")
    learner.resume = True
    learner.n_rounds = 2
    learner._history = []
    (tmp_path / "active_learning_history.json").write_text(
        json.dumps({
            "rounds": [{"round": 1, "early_stopped": True}],
            "early_stopped": True,
            "final_round": 1,
        }),
        encoding="utf-8",
    )

    _, start_round, _, early_stopped = learner._restore_progress(
        pd.DataFrame({"objective": [0.3]})
    )
    assert start_round == 3
    assert early_stopped


def test_active_learning_exports_best_lammps_verified_parameters(tmp_path):
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.output_dir = str(tmp_path)
    learner.param_names = ["q1", "q2"]
    learner.config = {"targets": {"density": {"value": 1.0}}}
    frame = pd.DataFrame([
        {
            "round": 0, "label": "bo", "success": True,
            "objective": 0.20, "q1": 0.1, "q2": -0.1,
            "calc_density": 0.9,
        },
        {
            "round": 1, "label": "al_round_1", "success": True,
            "objective": 0.08, "q1": 0.2, "q2": -0.2,
            "calc_density": 0.98,
        },
        {
            "round": 1, "label": "failed", "success": False,
            "objective": 0.01, "q1": 0.3, "q2": -0.3,
            "calc_density": 1.0,
        },
    ])

    document = learner._write_final_parameters(frame)
    path = tmp_path / "final_parameters.json"
    assert document["objective"] == 0.08
    assert document["source_label"] == "al_round_1"
    assert _load_parameters(path) == {"q1": 0.2, "q2": -0.2}
