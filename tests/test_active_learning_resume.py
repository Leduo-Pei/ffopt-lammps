import json

import pandas as pd

from engine.active_learning import ActiveLearner


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

