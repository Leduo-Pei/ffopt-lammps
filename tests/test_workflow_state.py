
from workflow.state import WorkflowState, canonical_hash


def test_state_persists_completed_stage_and_checks_artifacts(tmp_path):
    database = tmp_path / "state.sqlite"
    artifact = tmp_path / "bo" / "all_results.csv"
    signature = canonical_hash({"stage": "bo", "config": 1})

    with WorkflowState(database) as state:
        state.initialize({"project": "example"})
        record = state.prepare(
            "bo", signature, ["python", "-m", "engine.run"],
            artifact.parent, [artifact],
        )
        assert record.status == "pending"
        state.transition("bo", "running", increment_attempt=True)
        artifact.parent.mkdir()
        artifact.write_text("objective\n0.1\n", encoding="utf-8")
        state.transition("bo", "completed")
        assert state.is_complete("bo", signature)

    with WorkflowState(database) as state:
        assert state.metadata()["project"] == "example"
        assert state.get("bo").attempt == 1
        assert state.is_complete("bo", signature)
        artifact.unlink()
        assert not state.is_complete("bo", signature)


def test_changed_signature_resets_stage_attempt(tmp_path):
    database = tmp_path / "state.sqlite"
    with WorkflowState(database) as state:
        state.prepare("nn", "old", ["train"], tmp_path / "nn", [tmp_path / "a"])
        state.transition("nn", "running", increment_attempt=True)
        state.transition("nn", "failed", message="interrupted")
        record = state.prepare(
            "nn", "new", ["train", "--new"], tmp_path / "nn", [tmp_path / "b"]
        )
        assert record.status == "pending"
        assert record.attempt == 0
        assert record.job_id is None


def test_active_stage_preserves_submitted_command(tmp_path):
    database = tmp_path / "state.sqlite"
    with WorkflowState(database) as state:
        state.prepare(
            "bo", "same", ["python", "old_config.json"],
            tmp_path / "bo", [tmp_path / "result.csv"],
        )
        state.transition("bo", "waiting", job_id="123", increment_attempt=True)
        record = state.prepare(
            "bo", "same", ["python", "new_config.json"],
            tmp_path / "bo", [tmp_path / "result.csv"],
        )
        assert record.status == "waiting"
        assert record.command == ["python", "old_config.json"]

        state.transition("bo", "failed", message="timeout")
        record = state.prepare(
            "bo", "same", ["python", "new_config.json"],
            tmp_path / "bo", [tmp_path / "result.csv"],
        )
        assert record.command == ["python", "new_config.json"]
