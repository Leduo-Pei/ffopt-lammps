from unittest.mock import Mock

from engine.nn_surrogate import NNSurrogate


def test_empty_candidate_validation_does_not_start_executor(tmp_path):
    surrogate = NNSurrogate.__new__(NNSurrogate)
    surrogate.output_dir = str(tmp_path)
    surrogate.runner = Mock()
    surrogate.save_traj = False

    assert surrogate._validate_candidates([]) == []
    surrogate.runner.evaluate_batch.assert_not_called()
