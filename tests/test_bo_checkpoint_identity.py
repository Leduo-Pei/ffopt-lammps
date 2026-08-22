from __future__ import annotations

import copy
from pathlib import Path

import pytest

from engine import optimizer as optimizer_module
from engine.optimizer import CheckpointIdentityError, ForceFieldOptimizer
from workflow.pipeline import PipelineRunner
from workflow.project import Project


torch = optimizer_module.torch


def _optimizer_config() -> dict:
    return {
        "manifest": {"system_name": "demo"},
        "atom_types": [{
            "type": 1,
            "label": "X",
            "mass": 1.0,
            "params": {
                "sigma": {"min": 1.0, "max": 3.0, "init": 2.0},
            },
        }],
        "pair_params": {"mixing": "geometric", "explicit_pairs": []},
        "charge": {"enabled": False},
        "targets": {"a": {"value": 2.0, "tolerance": 0.02, "weight": 1.0}},
        "lammps": {"cutoff": 12.5, "executable": "lmp", "timeout": 60},
        "optimization": {
            "method": "turbo",
            "objective": "feasible_coverage",
            "coverage": {"archive_size": 20},
            "n_initial": 4,
            "n_bo_iterations": 10,
            "random_seed": 7,
        },
        "workflow": {"bo_stage_signature": "stage-a"},
    }


def _checkpoint_optimizer(path: Path, config: dict) -> ForceFieldOptimizer:
    optimizer = ForceFieldOptimizer.__new__(ForceFieldOptimizer)
    optimizer.config = copy.deepcopy(config)
    optimizer.param_space = [("X_sigma", 1.0, 3.0)]
    optimizer.param_names = ["X_sigma"]
    optimizer.bo_method = "turbo"
    optimizer.current_round = 3
    optimizer.train_X = torch.tensor([[2.0]], dtype=torch.double)
    optimizer.train_Y = torch.tensor([[0.25]], dtype=torch.double)
    optimizer.all_X = torch.tensor([[2.0]], dtype=torch.double)
    optimizer.all_feasible = [True]
    optimizer.all_results = [{"success": True, "objective": 0.25}]
    optimizer.best_history = [0.25]
    optimizer._best_valid_obj = 0.25
    optimizer.round_times = [1.0]
    optimizer.bo_finished = False
    optimizer._turbo_state = None
    optimizer.work_dir = str(path)
    optimizer.run_root = str(path.parent)
    optimizer.ckpt_dir = str(path / "checkpoints")
    Path(optimizer.ckpt_dir).mkdir(parents=True, exist_ok=True)
    optimizer._update_feasibility_classifier = lambda: None
    return optimizer


def test_checkpoint_with_same_identity_resumes_normally(tmp_path):
    config = _optimizer_config()
    writer = _checkpoint_optimizer(tmp_path / "run", config)
    writer._save_checkpoint()

    reader = _checkpoint_optimizer(tmp_path / "run", config)
    reader.current_round = 0
    reader.train_Y = torch.empty(0, 1, dtype=torch.double)

    assert reader.load_checkpoint(
        str(Path(writer.ckpt_dir) / "latest.json")
    )
    assert reader.current_round == 3
    assert reader.train_Y.tolist() == [[0.25]]


@pytest.mark.parametrize(
    "change",
    [
        lambda cfg: cfg["optimization"].__setitem__(
            "objective", "weighted_rmse"
        ),
        lambda cfg: cfg["optimization"]["coverage"].__setitem__(
            "archive_size", 40
        ),
        lambda cfg: cfg["targets"]["a"].__setitem__("value", 2.1),
        lambda cfg: cfg["lammps"].__setitem__("cutoff", 8.0),
        lambda cfg: cfg["workflow"].__setitem__(
            "bo_stage_signature", "stage-b"
        ),
    ],
    ids=["objective", "coverage", "target", "cutoff", "stage-signature"],
)
def test_checkpoint_rejects_scientific_or_optimization_drift(tmp_path, change):
    original = _optimizer_config()
    writer = _checkpoint_optimizer(tmp_path / "run", original)
    writer._save_checkpoint()

    changed = copy.deepcopy(original)
    change(changed)
    reader = _checkpoint_optimizer(tmp_path / "run", changed)
    reader.current_round = 0
    reader.train_Y = torch.empty(0, 1, dtype=torch.double)

    with pytest.raises(CheckpointIdentityError, match="different BO scientific"):
        reader.load_checkpoint(str(Path(writer.ckpt_dir) / "latest.json"))
    assert reader.current_round == 0
    assert reader.train_Y.numel() == 0


def test_legacy_checkpoint_is_rejected_before_loading_state(tmp_path):
    optimizer = _checkpoint_optimizer(tmp_path / "run", _optimizer_config())
    checkpoint = Path(optimizer.ckpt_dir) / "latest.json"
    checkpoint.write_text(
        '{"round": 99, "train_X": [[2.0]], "train_Y": [[9.0]]}',
        encoding="utf-8",
    )
    optimizer.current_round = 0

    with pytest.raises(CheckpointIdentityError, match="legacy checkpoints"):
        optimizer.load_checkpoint(str(checkpoint))
    assert optimizer.current_round == 0


def _pipeline_project(tmp_path: Path) -> Project:
    input_path = tmp_path / "ffopt.in"
    input_path.write_text("ffopt 1\nproject demo\n", encoding="utf-8")
    return Project(input_path, {
        "project": {"name": "demo", "run_root": "runs/demo"},
        "modules": {},
        "pipeline": {"stages": ["bo"]},
    })


def _pipeline_config() -> dict:
    return {
        "machine": {"backend": "local"},
        "manifest": {"system_name": "demo"},
        "lammps": {"executable": "lmp", "mpiexec": "mpiexec", "cutoff": 12.5},
        "optimization": {
            "method": "turbo",
            "objective": "feasible_coverage",
            "coverage": {"archive_size": 20},
        },
        "checkpoint": {"enabled": True, "directory": "checkpoints"},
        "nn": {"enabled": True},
        "active_learning": {"enabled": True},
    }


def test_pipeline_pins_resume_to_own_checkpoint_and_stage_signature(
    tmp_path, monkeypatch
):
    project = _pipeline_project(tmp_path)
    config = _pipeline_config()
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="local", run_id="trial", resume=True,
    )

    spec = runner.build_specs()[0]
    assert spec.command[spec.command.index("--stage-signature") + 1] == spec.signature
    checkpoint = spec.command[spec.command.index("--checkpoint") + 1]
    assert Path(checkpoint) == spec.output_dir / "checkpoints" / "latest.json"
    assert "--resume" in spec.command


@pytest.mark.parametrize("field", ["objective", "coverage"])
def test_pipeline_rejects_changed_bo_stage_signature(
    tmp_path, monkeypatch, field
):
    project = _pipeline_project(tmp_path)
    config = _pipeline_config()
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)

    def fake_run_local(self, state, spec):
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    first = PipelineRunner(project=project, machine="local", run_id="trial")
    assert first.run() == "completed"

    if field == "objective":
        config["optimization"]["objective"] = "weighted_rmse"
    else:
        config["optimization"]["coverage"]["archive_size"] = 40
    second = PipelineRunner(
        project=project, machine="local", run_id="trial", resume=True,
    )
    with pytest.raises(RuntimeError, match="BO stage signature changed.*--new"):
        second.run()
