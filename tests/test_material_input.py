from __future__ import annotations

from pathlib import Path

import pytest

from engine.parameter_space import build_parameter_space
from workflow.input_compiler import compile_input
from workflow.input_file import InputFileError, parse_input_file
from workflow.pipeline import PipelineRunner
from workflow.project import Project


ROOT = Path(__file__).resolve().parents[1]


def _input_text(
    *,
    material: str = "material elemental Fe",
    crystal: str = "crystal bcc",
    first_epsilon: float = 1.0,
    second_epsilon: float = 1.0,
    difference: str = "difference sigma Fe_corner Fe_body max 0.75 A",
    explicit_charge: bool = False,
    data: Path | None = None,
) -> str:
    data_path = data or (ROOT / "data" / "bulk" / "BTAH_822_bulk.data")
    charge = " 0.0" if explicit_charge else ""
    return f"""\
ffopt 1
project fe_test
{material}
{crystal}
workflow bo

parameters
    range epsilon absolute 0.001 10
    range sigma absolute 0.001 5
    mixing default
    tie epsilon all
    {difference}
    type 1 Fe_corner {first_epsilon} 2.30{charge}
    type 2 Fe_body   {second_epsilon} 2.50{charge}
end

property bulk
    data "{data_path.as_posix()}"
    cells_in_data 1 1 1
    target a 2.8665 A weight 1 tolerance 0.03
end
"""


def _write(tmp_path: Path, source: str, name: str = "ffopt.in") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _elemental_data(tmp_path: Path) -> Path:
    path = tmp_path / "fe.data"
    path.write_text(
        """BCC Fe

2 atoms
2 atom types

0 2.8665 xlo xhi
0 2.8665 ylo yhi
0 2.8665 zlo zhi

Masses

1 55.845 # Fe_corner
2 55.845 # Fe_body

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 2 0.0 1.43325 1.43325 1.43325
""",
        encoding="utf-8",
    )
    return path


def _compile_packaged_fe_input(source: Path, tmp_path: Path, name: str):
    data = _elemental_data(tmp_path)
    text = source.read_text(encoding="utf-8")
    for filename in (
        "Fe_a_2type_555.data",
        "Fe_a_2type_555_complete.data",
        "Fe_a_2type_555_split.data",
    ):
        text = text.replace(f"data/{filename}", data.as_posix())
    return compile_input(parse_input_file(_write(tmp_path, text, name)))


def test_packaged_fe_full_and_non_scientific_canary_compile(tmp_path):
    example_dir = ROOT / "examples" / "fe_bcc"
    full = _compile_packaged_fe_input(example_dir / "ffopt.in", tmp_path, "full.in")
    canary_path = example_dir / "ffopt.canary.in"
    canary = _compile_packaged_fe_input(canary_path, tmp_path, "canary.in")

    assert full.config["optimization"]["stability_audit"]["enabled"] is False
    assert canary.document.project == "fe_bcc_canary"
    assert "NON_SCIENTIFIC_CANARY" in canary_path.read_text(encoding="utf-8")
    assert canary.config["optimization"]["n_initial"] == 4
    assert canary.config["optimization"]["n_bo_iterations"] == 1
    assert canary.project_data["stages"]["sample"]["n_points"] == 8
    assert canary.project_data["pipeline"]["stage_repetitions"] == {
        "constrained_al": 2
    }
    assert canary.project_data["stages"]["screen"]["maximum"] == 12
    assert canary.project_data["stages"]["finalists"]["maximum"] == 2
    assert canary.config["elasticity"]["modules"]["dynamic"]["protocol"][
        "seeds"
    ] == [101]
    assert canary.config["elasticity"]["modules"]["dynamic"][
        "validation_protocol"
    ]["seeds"] == [404]


def test_elemental_material_compiles_charge_defaults_and_parameter_graph(tmp_path):
    compiled = compile_input(parse_input_file(_write(
        tmp_path,
        _input_text(data=_elemental_data(tmp_path)),
    )))

    assert compiled.material == {
        "kind": "elemental",
        "name": "Fe",
        "force_field": "lj/cut",
        "pair_style": "lj/cut",
    }
    assert compiled.crystal == {"family": "bcc"}
    assert compiled.dimensions == 3
    assert compiled.config["charge"]["enabled"] is False
    assert compiled.config["manifest"]["crystal_structure"] == "bcc"
    assert compiled.config["pair_params"]["mixing_rule"] == "default"
    assert compiled.config["material"] == compiled.material
    assert compiled.config["crystal"] == compiled.crystal
    assert compiled.config["parameter_constraints"] == {
        "ties": [{
            "parameter": "epsilon",
            "types": ["Fe_corner", "Fe_body"],
            "source": "Fe_corner",
        }],
        "differences": [{
            "parameter": "sigma",
            "types": ["Fe_corner", "Fe_body"],
            "max_abs": 0.75,
            "unit": "A",
        }],
    }
    assert compiled.config["pair_params"]["derived_params"] == [{
        "target": "Fe_body_epsilon",
        "expression": "Fe_corner_epsilon",
        "comment": "tie epsilon all",
    }]
    charges = [item["params"]["charge"] for item in compiled.config["atom_types"]]
    assert charges == [0.0, 0.0]
    assert [name for name, _, _ in build_parameter_space(compiled.config)] == [
        "Fe_corner_epsilon",
        "Fe_corner_sigma",
        "Fe_body_sigma",
    ]


def test_legacy_molecular_input_keeps_extension_keys_out_of_engine_config():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    )

    assert compiled.material == {"kind": "molecular", "name": "btah_charge_only"}
    assert compiled.crystal == {"family": "molecular"}
    assert compiled.parameter_constraints == {"ties": [], "differences": []}
    assert "material" not in compiled.config
    assert "crystal" not in compiled.config
    assert "parameter_constraints" not in compiled.config
    assert compiled.project_data["pipeline"] == {
        "stages": compiled.document.workflow,
        "audit": compiled.project_data["stages"]["audit"],
    }


def test_elemental_public_workflow_compiles_to_repeated_constrained_al(tmp_path):
    source = _input_text(data=_elemental_data(tmp_path)).replace(
        "workflow bo",
        "workflow bo sample audit screen nn al finalists validate",
    )
    source += """
bo
    objective feasible_coverage
    coverage archive 96 pool 16384 feasible 0.50 boundary 0.25 uncertainty 0.15 global 0.10
end

screen
    minimum 480
    per_dimension 160
    maximum 1200
end

al
    acquisition constrained_minimax
    rounds 8
    candidates 76
    static_candidates 38
    candidate_pool 65536
    improvement_fraction 0.50
    boundary_fraction 0.30
    global_fraction 0.20
    early_stop patience 3
    early_stop improvement 0.5
end

finalists
    minimum 10
    maximum 20
    window 1.0
    diverse 1
end
"""
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert compiled.document.workflow == [
        "bo", "sample", "audit", "screen", "nn", "al", "finalists", "validate",
    ]
    assert compiled.project_data["pipeline"] == {
        "stages": [
            "bo", "sample", "audit", "screen", "nn", "constrained_al",
            "finalists", "validate",
        ],
        "audit": {"top_k": 8, "seeds": [101, 202, 303]},
        "stage_repetitions": {
            "constrained_al": compiled.config["active_learning"]["n_rounds"],
        },
        "candidates": {
            "minimum": 480,
            "per_dimension": 160,
            "maximum": 1200,
            "core_fraction": 0.75,
            "buffer_multiplier": 1.50,
            "objective_elite_fraction": 0.10,
        },
        "static": {
            "minimum": 480,
            "per_dimension": 160,
            "maximum": 1200,
            "core_fraction": 0.75,
            "buffer_multiplier": 1.50,
            "objective_elite_fraction": 0.10,
        },
        "constrained_al": {
            "maximum_rounds": 8,
            "patience": 3,
            "minimum_improvement_percent_points": 0.5,
            "structural_proposals_per_round": 76,
            "mechanical_proposals_per_round": 38,
            "improvement_fraction": 0.50,
            "feasible_fraction": 0.50,
            "boundary_fraction": 0.30,
            "global_fraction": 0.20,
            "candidate_pool": 65536,
            "seed": 42,
            "structural_seeds": [101, 202, 303],
        },
        "finalists": {
            "minimum": 10,
            "maximum": 20,
            "near_optimal_window_percent": 1.0,
            "diverse_reserve": 1,
            "top_n": 20,
            "diversity_slots": 1,
        },
        "graph_mode": "linear",
    }
    assert compiled.config["optimization"]["objective"] == "feasible_coverage"
    assert compiled.config["optimization"]["method"] == "gp"
    assert compiled.config["optimization"]["coverage"] == {
        "archive_target": 96,
        "candidate_pool": 16384,
        "feasible_fraction": 0.50,
        "boundary_fraction": 0.25,
        "uncertainty_fraction": 0.15,
        "global_fraction": 0.10,
    }
    assert compiled.config["active_learning"]["acquisition"] == "constrained_minimax"
    assert compiled.config["active_learning"]["early_stop"] == {
        "patience": 3,
        "min_improvement": 0.5,
    }
    assert compiled.project_data["stages"]["finalists"]["maximum"] == 20


def test_compiled_material_stage_graph_consumes_public_stage_settings(
    tmp_path, monkeypatch
):
    source = _input_text(data=_elemental_data(tmp_path)).replace(
        "workflow bo",
        "workflow bo sample audit screen nn al finalists validate",
    )
    source += """
audit
    top_k 6
    seeds 11 13 17
end

screen
    minimum 12
    per_dimension 7
    maximum 24
    core_fraction 0.6
    buffer_multiplier 1.25
    elite_fraction 0.2
end

al
    acquisition constrained_minimax
    rounds 4
    candidates 9
    static_candidates 5
    candidate_pool 111
    improvement_fraction 0.4
    boundary_fraction 0.35
    global_fraction 0.25
    early_stop patience 2
    early_stop improvement 0.125
end

finalists
    minimum 3
    maximum 7
    window 2.5
    diverse 2
end
"""
    path = _write(tmp_path, source)
    compiled = compile_input(parse_input_file(path))
    project = Project(
        path,
        compiled.project_data,
        runtime_config=compiled.config,
        compilation=compiled,
    )
    monkeypatch.setattr(
        "workflow.pipeline.build_refinement_spec",
        lambda *_args, **_kwargs: {"schema": "test-refinement-settings"},
    )
    runner = PipelineRunner(
        project=project,
        machine="local",
        run_id="compiled-settings-contract",
        dry_run=True,
    )

    assert runner.stage_names == [
        "bo", "sample", "audit", "candidates", "static", "nn",
        "constrained_al_01", "constrained_al_02", "constrained_al_03",
        "constrained_al_04", "finalists", "validate",
    ]
    expected_screen = {
        "minimum": 12,
        "per_dimension": 7,
        "maximum": 24,
        "core_fraction": 0.6,
        "buffer_multiplier": 1.25,
        "objective_elite_fraction": 0.2,
    }
    assert {
        key: runner._stage_settings("candidates")[key]
        for key in expected_screen
    } == expected_screen
    assert {
        key: runner._stage_settings("static")[key]
        for key in expected_screen
    } == expected_screen

    refinement = runner._stage_settings("constrained_al_01")
    assert {
        key: refinement[key]
        for key in (
            "maximum_rounds",
            "patience",
            "minimum_improvement_percent_points",
            "structural_proposals_per_round",
            "mechanical_proposals_per_round",
            "improvement_fraction",
            "feasible_fraction",
            "boundary_fraction",
            "global_fraction",
            "candidate_pool",
            "seed",
            "structural_seeds",
        )
    } == {
        "maximum_rounds": 4,
        "patience": 2,
        "minimum_improvement_percent_points": 0.125,
        "structural_proposals_per_round": 9,
        "mechanical_proposals_per_round": 5,
        "improvement_fraction": 0.4,
        "feasible_fraction": 0.4,
        "boundary_fraction": 0.35,
        "global_fraction": 0.25,
        "candidate_pool": 111,
        "seed": 42,
        "structural_seeds": [11, 13, 17],
    }
    assert runner._stage_settings("finalists") == {
        "minimum": 3,
        "maximum": 7,
        "near_optimal_window_percent": 2.5,
        "diverse_reserve": 2,
        "top_n": 7,
        "diversity_slots": 2,
    }

    specs = {item.name: item for item in runner.build_specs()}
    candidates = specs["candidates"].command
    assert candidates[candidates.index("--screen-minimum") + 1] == "12"
    assert candidates[candidates.index("--screen-maximum") + 1] == "24"
    assert candidates[candidates.index("--screen-core-fraction") + 1] == "0.6"
    refinement_command = specs["constrained_al_01"].command
    assert refinement_command[refinement_command.index("--seed") + 1] == "42"
    seed_index = refinement_command.index("--structural-seeds")
    assert refinement_command[seed_index + 1:seed_index + 4] == [
        "11", "13", "17",
    ]
    finalists = specs["finalists"].command
    assert finalists[finalists.index("--top-n") + 1] == "7"
    assert finalists[
        finalists.index("--near-optimal-window-percent") + 1
    ] == "2.5"
    assert finalists[finalists.index("--diversity-slots") + 1] == "2"


def test_feasible_coverage_without_detail_line_persists_defaults(tmp_path):
    source = _input_text(data=_elemental_data(tmp_path)) + """
bo
    objective feasible_coverage
end
"""

    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert compiled.config["optimization"]["coverage"] == {
        "archive_target": 96,
        "candidate_pool": 16384,
        "feasible_fraction": 0.50,
        "boundary_fraction": 0.25,
        "uncertainty_fraction": 0.15,
        "global_fraction": 0.10,
    }


def test_elemental_workflow_rejects_legacy_and_out_of_order_stages(tmp_path):
    data = _elemental_data(tmp_path)
    legacy = _input_text(data=data).replace(
        "workflow bo", "workflow bo audit finalize",
    )
    with pytest.raises(InputFileError, match="material workflow does not support"):
        parse_input_file(_write(tmp_path, legacy, "legacy.in"))

    out_of_order = _input_text(data=data).replace(
        "workflow bo", "workflow bo sample screen audit",
    )
    with pytest.raises(InputFileError, match="workflow stages must follow"):
        parse_input_file(_write(tmp_path, out_of_order, "order.in"))


def test_packaged_fe_bcc_example_compiles_as_one_managed_pipeline(tmp_path):
    source = (ROOT / "examples" / "fe_bcc" / "ffopt.in").read_text(
        encoding="utf-8"
    )
    replacements = {}
    data_text = _elemental_data(tmp_path).read_text(encoding="utf-8")
    for name in (
        "Fe_a_2type_555.data",
        "Fe_a_2type_555_complete.data",
        "Fe_a_2type_555_split.data",
    ):
        path = tmp_path / name
        path.write_text(data_text, encoding="utf-8")
        replacements[f"data/{name}"] = path.as_posix()
    for old, new in replacements.items():
        source = source.replace(old, new)

    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert compiled.dimensions == 3
    assert compiled.project_data["pipeline"]["stage_repetitions"] == {
        "constrained_al": 8,
    }
    assert compiled.config["elasticity"]["modules"]["dynamic"]["protocol"][
        "seeds"
    ] == [101, 202, 303]


@pytest.mark.parametrize("kind", ["molecular", "multicomponent"])
def test_explicit_non_elemental_material_kinds_compile(tmp_path, kind):
    source = _input_text(
        material=f"material {kind} example",
        crystal="crystal molecular",
        explicit_charge=True,
    ).replace("workflow bo", "workflow validate")
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert compiled.material == {"kind": kind, "name": "example"}
    assert compiled.crystal == {"family": "molecular"}
    assert compiled.config["charge"]["enabled"] is True


@pytest.mark.parametrize("kind", ["molecular", "multicomponent"])
def test_non_elemental_atom_types_still_require_explicit_charge(tmp_path, kind):
    source = _input_text(material=f"material {kind}")
    path = _write(tmp_path, source)
    expected_line = next(
        number
        for number, line in enumerate(source.splitlines(), 1)
        if "type 1 Fe_corner" in line
    )

    with pytest.raises(InputFileError, match="charge may be omitted") as captured:
        parse_input_file(path)
    assert captured.value.line == expected_line


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            "material elemental Fe\nmaterial elemental W",
            "duplicate material declaration",
        ),
        ("crystal bcc\ncrystal molecular", "duplicate crystal declaration"),
        ("tie epsilon all\n    tie epsilon all", "duplicate tie"),
        (
            "difference sigma Fe_corner Fe_body max 0.75 A\n"
            "    difference sigma Fe_body Fe_corner max 0.50 A",
            "duplicate difference",
        ),
    ],
)
def test_material_parameter_declarations_reject_duplicates(
    tmp_path, replacement, message
):
    source = _input_text()
    if replacement.startswith("material"):
        source = source.replace("material elemental Fe", replacement)
    elif replacement.startswith("crystal"):
        source = source.replace("crystal bcc", replacement)
    elif replacement.startswith("tie"):
        source = source.replace("tie epsilon all", replacement)
    else:
        source = source.replace(
            "difference sigma Fe_corner Fe_body max 0.75 A", replacement
        )

    with pytest.raises(InputFileError, match=message):
        parse_input_file(_write(tmp_path, source))


def test_difference_unknown_type_reports_declaration_line(tmp_path):
    source = _input_text(
        difference="difference sigma Fe_corner missing max 0.75 A"
    )
    expected_line = next(
        number
        for number, line in enumerate(source.splitlines(), 1)
        if "difference sigma" in line
    )

    with pytest.raises(InputFileError, match="unknown type label") as captured:
        parse_input_file(_write(tmp_path, source))
    assert captured.value.line == expected_line


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("range sigma absolute 0.001 5", "range sigma absolute 5 0.001", "lower bound"),
        (
            "difference sigma Fe_corner Fe_body max 0.75 A",
            "difference sigma Fe_corner Fe_body max 0 A",
            "must be positive",
        ),
        (
            "difference sigma Fe_corner Fe_body max 0.75 A",
            "difference sigma Fe_corner Fe_body max 0.75 nm",
            "unit must be",
        ),
    ],
)
def test_invalid_material_parameter_ranges_report_exact_line(
    tmp_path, old, new, message
):
    source = _input_text().replace(old, new)
    expected_line = next(
        number for number, line in enumerate(source.splitlines(), 1) if new in line
    )

    with pytest.raises(InputFileError, match=message) as captured:
        parse_input_file(_write(tmp_path, source))
    assert captured.value.line == expected_line


def test_tie_and_difference_initial_value_errors_use_constraint_lines(tmp_path):
    data = _elemental_data(tmp_path)
    tied_source = _input_text(second_epsilon=1.1, data=data)
    tie_line = next(
        number
        for number, line in enumerate(tied_source.splitlines(), 1)
        if "tie epsilon" in line
    )
    with pytest.raises(InputFileError, match="identical initial") as captured:
        compile_input(parse_input_file(_write(tmp_path, tied_source, "tie.in")))
    assert captured.value.line == tie_line

    difference_source = _input_text(
        difference="difference sigma Fe_corner Fe_body max 0.10 A",
        data=data,
    )
    difference_line = next(
        number
        for number, line in enumerate(difference_source.splitlines(), 1)
        if "difference sigma" in line
    )
    with pytest.raises(InputFileError, match="initial sigma difference") as captured:
        compile_input(
            parse_input_file(_write(tmp_path, difference_source, "difference.in"))
        )
    assert captured.value.line == difference_line
