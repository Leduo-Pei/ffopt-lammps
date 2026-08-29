from __future__ import annotations

from pathlib import Path

import pytest

from workflow.artifact_manifest import scientific_config_hash
from workflow.input_compiler import compile_input
from workflow.input_file import InputFileError, parse_input_file
from workflow.pipeline import _scientific_config


def _bcc_data() -> str:
    return """BCC Fe test box

2 atoms
2 atom types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 55.845 # Fe_corner
2 55.845 # Fe_body

Pair Coeffs # lj/cut

1 6.0 2.3 # Fe_corner
2 6.0 2.3 # Fe_body

Atoms # full

1 1 1 0.0 0.0 0.0 0.0 0 0 0
2 1 2 0.0 1.43325 1.43325 1.43325 0 0 0
"""


def _elasticity_block(*, dynamic: bool = True) -> str:
    dynamic_text = """
    module dynamic promotion
    target dynamic B 166.2 GPa
    target dynamic Cprime 48.15 GPa
    target dynamic C44 115.87 GPa
    temperature 300 K
    timestep 1 fs
    npt_equilibration 21000
    nvt_equilibration 20000
    production 40000
    seeds 101 202 303
    validation_strain 0.001 0.003
    validation_npt_equilibration 200000
    validation_nvt_equilibration 50000
    validation_production 500000
    validation_seeds 404 505 606
""" if dynamic else ""
    return f"""property elasticity
    module static objective
    target static B 173.1 GPa
    target static Cprime 52.5 GPa
    target static C44 121.9 GPa
{dynamic_text}    gate lattice 1 percent
    gate angles 1 degree
    gate density 1 percent
    gate surface 5 percent
    born required
    r2 0.98
    tier 20 percent
    strain 0.002 0.004 0.006
    replicate 2 2 2
end
"""


def _isotropic_elasticity_block(*, dynamic: bool = True) -> str:
    block = _elasticity_block(dynamic=dynamic)
    replacements = {
        "    target static B 173.1 GPa\n"
        "    target static Cprime 52.5 GPa\n"
        "    target static C44 121.9 GPa\n":
        "    target static B 173.1 GPa\n"
        "    target static G 86.94 GPa\n"
        "    target static E 223.4 GPa\n"
        "    target static nu 0.2848 1\n",
        "    target dynamic B 166.2 GPa\n"
        "    target dynamic Cprime 48.15 GPa\n"
        "    target dynamic C44 115.87 GPa\n":
        "    target dynamic B 170.0 GPa\n"
        "    target dynamic G 82.0 GPa\n"
        "    target dynamic E 211.0 GPa\n"
        "    target dynamic nu 0.29 1\n",
    }
    for old, new in replacements.items():
        block = block.replace(old, new)
    return block


def _project_text(elasticity: str | None = None) -> str:
    return f"""ffopt 1
project fe_elastic
material elemental Fe
crystal bcc
workflow bo validate

parameters
    range epsilon absolute 0.001 10
    range sigma absolute 0.001 5
    cutoff 12.5 A
    mixing default
    tie epsilon all
    type 1 Fe_corner 6.0 2.3
    type 2 Fe_body 6.0 2.3
end

property bulk
    data data/bulk.data
    cells_in_data 1 1 1
    target a 2.8665 A tolerance 0.03
    target b 2.8665 A tolerance 0.03
    target c 2.8665 A tolerance 0.03
    target alpha 90 degree tolerance 1
    target beta 90 degree tolerance 1
    target gamma 90 degree tolerance 1
    target density 7.874 g/cm3 tolerance 0.08
end

property surface
    complete data/complete.data
    split data/split.data
    facet 110
    target 2.34 J/m2 tolerance 0.117
end

{elasticity or _elasticity_block()}
"""


def _write(tmp_path: Path, source: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for name in ("bulk.data", "complete.data", "split.data"):
        (data_dir / name).write_text(_bcc_data(), encoding="utf-8")
    path = tmp_path / "ffopt.in"
    path.write_text(source, encoding="utf-8")
    return path


def test_cubic_elasticity_contract_compiles_without_polluting_legacy_targets(
    tmp_path: Path,
) -> None:
    compiled = compile_input(parse_input_file(_write(tmp_path, _project_text())))
    config = compiled.config
    elasticity = config["elasticity"]

    assert compiled.fitted_properties == ("bulk", "surface", "elasticity")
    assert not {"B", "Cprime", "C44"} & set(config["targets"])
    assert elasticity["target_basis"] == ["B", "Cprime", "C44"]
    assert elasticity["derived_diagnostics_only"] == [
        "G_hill", "E_hill", "nu_hill"
    ]
    assert elasticity["derived_diagnostics_only_by_fidelity"] == {
        "static": ["G_hill", "E_hill", "nu_hill"],
        "dynamic": ["G_hill", "E_hill", "nu_hill"],
    }
    assert elasticity["selection"] == {
        "method": "constrained_minimax_relative_error",
        "structural_gates": {
            "lattice": {
                "maximum_relative_error_percent": 1.0,
                "unit": "percent",
            },
            "angles": {
                "maximum_absolute_error_degree": 1.0,
                "unit": "degree",
            },
            "density": {
                "maximum_relative_error_percent": 1.0,
                "unit": "percent",
            },
            "surface": {
                "maximum_relative_error_percent": 5.0,
                "unit": "percent",
            },
        },
        "fit_quality": {
            "minimum_r2": 0.98,
            "maximum_static_drift_percent": 5.0,
        },
        "born_stability": {"required": True},
    }
    assert elasticity["reporting"] == {
        "mechanical_tier_percent": 20.0,
        "tier_is_hard_gate": False,
    }

    static = elasticity["modules"]["static"]
    assert static["role"] == "objective"
    assert static["fidelity"] == "static_0k"
    assert static["targets"] == {
        "B": {"value": 173.1, "unit": "GPa"},
        "Cprime": {"value": 52.5, "unit": "GPa"},
        "C44": {"value": 121.9, "unit": "GPa"},
    }
    assert static["protocol"]["strain_magnitudes"] == [0.002, 0.004, 0.006]
    assert static["protocol"]["replicate"] == [2, 2, 2]

    dynamic = elasticity["modules"]["dynamic"]
    assert dynamic["role"] == "promotion"
    assert dynamic["fidelity"] == "dynamic_300k"
    assert dynamic["targets"] == {
        "B": {"value": 166.2, "unit": "GPa"},
        "Cprime": {"value": 48.15, "unit": "GPa"},
        "C44": {"value": 115.87, "unit": "GPa"},
    }
    assert dynamic["protocol"]["temperature_k"] == 300.0
    assert dynamic["protocol"]["equilibration_steps"] == 21000
    assert dynamic["protocol"]["nvt_equilibration_steps"] == 20000
    assert dynamic["protocol"]["seeds"] == [101, 202, 303]
    assert dynamic["validation_protocol"] == {
        "strain_magnitudes": [0.001, 0.003],
        "equilibration_steps": 200000,
        "nvt_equilibration_steps": 50000,
        "production_steps": 500000,
        "seeds": [404, 505, 606],
    }
    assert "elasticity" not in config["property_evaluators"]
    assert "elasticity" not in config["validation"]["property_evaluators"]


def test_static_only_contract_uses_deterministic_scientific_defaults(
    tmp_path: Path,
) -> None:
    source = _project_text(_elasticity_block(dynamic=False))
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))
    elasticity = compiled.config["elasticity"]

    assert list(elasticity["modules"]) == ["static"]
    assert elasticity["modules"]["static"]["protocol"] == {
        "method": "symmetric_static_stress_zero_limit",
        "strain_magnitudes": [0.002, 0.004, 0.006],
        "replicate": [2, 2, 2],
        "temperature_k": 0.0,
        "energy_curvature": "diagnostic_only",
        "energy_stress_consistency_percent": 10.0,
        "maximum_zero_strain_extrapolation_drift_percent": 5.0,
        "reference_geometry_relative_tolerance": 1.0e-8,
        "reference_max_residual_pressure_gpa": 0.1,
        "reference_max_deviatoric_stress_gpa": 0.1,
    }

    baseline_hash = scientific_config_hash(_scientific_config(compiled.config))
    changed = source.replace("tier 20 percent", "tier 15 percent")
    changed_compiled = compile_input(parse_input_file(_write(tmp_path, changed)))
    changed_hash = scientific_config_hash(
        _scientific_config(changed_compiled.config)
    )
    assert changed_hash != baseline_hash


def test_isotropic_moduli_basis_compiles_with_dimensionless_poisson_target(
    tmp_path: Path,
) -> None:
    source = _project_text(_isotropic_elasticity_block())
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))
    elasticity = compiled.config["elasticity"]

    assert elasticity["target_basis"] == ["B", "G", "E", "nu"]
    assert elasticity["target_basis_by_fidelity"] == {
        "static": ["B", "G", "E", "nu"],
        "dynamic": ["B", "G", "E", "nu"],
    }
    assert elasticity["derived_diagnostics_only"] == [
        "Cprime", "C44", "zener_anisotropy"
    ]
    assert elasticity["derived_diagnostics_only_by_fidelity"] == {
        "static": ["Cprime", "C44", "zener_anisotropy"],
        "dynamic": ["Cprime", "C44", "zener_anisotropy"],
    }
    assert elasticity["modules"]["static"]["targets"] == {
        "B": {"value": 173.1, "unit": "GPa"},
        "G": {"value": 86.94, "unit": "GPa"},
        "E": {"value": 223.4, "unit": "GPa"},
        "nu": {"value": 0.2848, "unit": "1"},
    }
    assert elasticity["modules"]["dynamic"]["targets"] == {
        "B": {"value": 170.0, "unit": "GPa"},
        "G": {"value": 82.0, "unit": "GPa"},
        "E": {"value": 211.0, "unit": "GPa"},
        "nu": {"value": 0.29, "unit": "1"},
    }
    consistency = elasticity["target_consistency_by_fidelity"]
    assert set(consistency) == {"static", "dynamic"}
    assert consistency["static"]["status"] == (
        "consistent_within_reference_tolerance"
    )
    assert consistency["dynamic"]["maximum_relative_closure_error_percent"] < 1.0


def test_auxetic_poisson_target_is_valid_but_zero_is_not(tmp_path: Path) -> None:
    auxetic = _isotropic_elasticity_block(dynamic=False).replace(
        "target static nu 0.2848 1", "target static nu -0.20 dimensionless"
    )
    compiled = compile_input(parse_input_file(_write(tmp_path, _project_text(auxetic))))
    assert compiled.config["elasticity"]["modules"]["static"]["targets"]["nu"] == {
        "value": -0.2,
        "unit": "1",
    }
    assert compiled.config["elasticity"]["target_consistency_by_fidelity"]["static"][
        "status"
    ] == "inconsistent_experimental_targets"

    zero = auxetic.replace("target static nu -0.20 dimensionless", "target static nu 0 1")
    with pytest.raises(InputFileError, match="cannot be zero"):
        parse_input_file(_write(tmp_path, _project_text(zero)))


def test_target_basis_is_independent_per_fidelity(tmp_path: Path) -> None:
    block = _isotropic_elasticity_block().replace(
        "    target dynamic B 170.0 GPa\n"
        "    target dynamic G 82.0 GPa\n"
        "    target dynamic E 211.0 GPa\n"
        "    target dynamic nu 0.29 1\n",
        "    target dynamic B 166.2 GPa\n"
        "    target dynamic Cprime 48.15 GPa\n"
        "    target dynamic C44 115.87 GPa\n",
    )
    compiled = compile_input(parse_input_file(_write(tmp_path, _project_text(block))))
    elasticity = compiled.config["elasticity"]

    assert elasticity["target_basis_by_fidelity"] == {
        "static": ["B", "G", "E", "nu"],
        "dynamic": ["B", "Cprime", "C44"],
    }
    assert elasticity["derived_diagnostics_only_by_fidelity"] == {
        "static": ["Cprime", "C44", "zener_anisotropy"],
        "dynamic": ["G_hill", "E_hill", "nu_hill"],
    }


def test_legacy_dynamic_input_without_holdout_seeds_keeps_old_shape(
    tmp_path: Path,
) -> None:
    source = _project_text()
    for line in (
        "    validation_strain 0.001 0.003\n",
        "    validation_npt_equilibration 200000\n",
        "    validation_nvt_equilibration 50000\n",
        "    validation_production 500000\n",
        "    validation_seeds 404 505 606\n",
    ):
        source = source.replace(line, "")
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert "validation_protocol" not in (
        compiled.config["elasticity"]["modules"]["dynamic"]
    )


def test_elasticity_rejects_unknown_target_alias(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "target static B 173.1 GPa",
        "target static K 173.1 GPa",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="complete B/Cprime/C44 basis") as exc:
        parse_input_file(path)
    assert "target static K" in path.read_text().splitlines()[exc.value.line - 1]


def test_elasticity_rejects_mixed_target_bases(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "target static Cprime 52.5 GPa",
        "target static G 52.5 GPa",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="one complete target basis"):
        parse_input_file(path)


def test_dynamic_module_requires_its_own_complete_target_triplet(
    tmp_path: Path,
) -> None:
    block = _elasticity_block().replace(
        "    target dynamic C44 115.87 GPa\n", ""
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match=r"dynamic module requires exactly one complete"):
        parse_input_file(path)


def test_target_requires_matching_module(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    gate lattice 1 percent\n",
        "    target dynamic B 166.2 GPa\n"
        "    target dynamic Cprime 48.15 GPa\n"
        "    target dynamic C44 115.87 GPa\n"
        "    gate lattice 1 percent\n",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="dynamic target requires 'module dynamic"):
        parse_input_file(path)


def test_dynamic_only_settings_require_dynamic_module(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    gate lattice 1 percent\n",
        "    temperature 300 K\n    gate lattice 1 percent\n",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="temperature requires a dynamic module"):
        parse_input_file(path)


def test_dynamic_strain_requires_dynamic_module(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    gate lattice 1 percent\n",
        "    dynamic_strain 0.002 0.004\n    gate lattice 1 percent\n",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="dynamic_strain requires a dynamic module"):
        parse_input_file(path)


def test_static_drift_is_public_and_part_of_static_scientific_identity(
    tmp_path: Path,
) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    r2 0.98\n",
        "    r2 0.98\n    static_drift 2.5 percent\n",
    )
    compiled = compile_input(parse_input_file(_write(tmp_path, _project_text(block))))
    elasticity = compiled.config["elasticity"]

    assert elasticity["selection"]["fit_quality"] == {
        "minimum_r2": 0.98,
        "maximum_static_drift_percent": 2.5,
    }
    assert elasticity["modules"]["static"]["protocol"][
        "maximum_zero_strain_extrapolation_drift_percent"
    ] == 2.5


@pytest.mark.parametrize(
    "line",
    [
        "static_drift 0 percent",
        "static_drift -1 percent",
        "static_drift 5 GPa",
        "static_drift 5",
    ],
)
def test_static_drift_rejects_nonpositive_values_and_wrong_units(
    tmp_path: Path, line: str
) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    r2 0.98\n",
        f"    r2 0.98\n    {line}\n",
    )

    with pytest.raises(InputFileError, match="static_drift"):
        parse_input_file(_write(tmp_path, _project_text(block)))


def test_legacy_and_fidelity_specific_strains_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    block = _elasticity_block().replace(
        "    strain 0.002 0.004 0.006\n",
        "    strain 0.002 0.004 0.006\n"
        "    static_strain 0.0005 0.001 0.002\n",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="legacy strain and static_strain"):
        parse_input_file(path)


def test_static_strain_requires_third_magnitude_for_independent_drift_audit(
    tmp_path: Path,
) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    strain 0.002 0.004 0.006\n",
        "    static_strain 0.0005 0.001\n",
    )

    with pytest.raises(InputFileError, match="static_strain requires at least three"):
        parse_input_file(_write(tmp_path, _project_text(block)))


def test_finalists_require_dynamic_promotion_module(tmp_path: Path) -> None:
    workflow = "workflow bo sample audit screen nn al finalists validate"
    no_dynamic = _project_text(_elasticity_block(dynamic=False)).replace(
        "workflow bo validate", workflow
    )
    with pytest.raises(
        InputFileError, match="finalists.*module dynamic promotion"
    ):
        parse_input_file(_write(tmp_path, no_dynamic))


def test_finalists_reject_validation_only_dynamic_role(tmp_path: Path) -> None:
    workflow = "workflow bo sample audit screen nn al finalists validate"
    validation_only = _project_text().replace(
        "workflow bo validate", workflow
    ).replace("module dynamic promotion", "module dynamic validation")
    with pytest.raises(
        InputFileError, match="finalists.*role promotion"
    ):
        parse_input_file(_write(tmp_path, validation_only))


def test_validation_seeds_require_dynamic_module(tmp_path: Path) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "    gate lattice 1 percent\n",
        "    validation_seeds 404 505 606\n    gate lattice 1 percent\n",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(
        InputFileError, match="validation_seeds requires a dynamic module"
    ):
        parse_input_file(path)


def test_validation_seeds_must_be_independent_of_promotion(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _project_text().replace(
            "validation_seeds 404 505 606",
            "validation_seeds 303 404 505",
        ),
    )

    with pytest.raises(
        InputFileError,
        match=r"validation_seeds must be disjoint.*overlap=\[303\]",
    ):
        parse_input_file(path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("module static promotion", "static module role must be objective"),
        ("module dynamic objective", "dynamic module role must be promotion or validation"),
        ("r2 1.1", r"r2 must be in \(0, 1\]"),
        ("strain 0.004 0.002", "strictly increasing"),
        ("seeds 101 101", "seeds must be unique"),
        (
            "validation_seeds 404 404",
            "validation_seeds must be unique",
        ),
        ("tier 0 percent", "reporting tier must be positive"),
    ],
)
def test_elasticity_setting_errors_are_explicit(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    substitutions = {
        "module static promotion": ("module static objective", replacement),
        "module dynamic objective": ("module dynamic promotion", replacement),
        "r2 1.1": ("r2 0.98", replacement),
        "strain 0.004 0.002": ("strain 0.002 0.004 0.006", replacement),
        "seeds 101 101": ("seeds 101 202 303", replacement),
        "validation_seeds 404 404": (
            "validation_seeds 404 505 606",
            replacement,
        ),
        "tier 0 percent": ("tier 20 percent", replacement),
    }
    old, new = substitutions[replacement]
    path = _write(tmp_path, _project_text().replace(old, new))

    with pytest.raises(InputFileError, match=message):
        parse_input_file(path)


def test_gate_requires_corresponding_structural_target(tmp_path: Path) -> None:
    source = _project_text().replace(
        "    target gamma 90 degree tolerance 1\n", ""
    )
    path = _write(tmp_path, source)
    gate_line = next(
        index
        for index, line in enumerate(path.read_text().splitlines(), 1)
        if "gate angles" in line
    )

    with pytest.raises(InputFileError, match="missing=.*gamma_ang") as exc:
        compile_input(parse_input_file(path))
    assert exc.value.line == gate_line


def test_elasticity_weight_option_is_rejected_instead_of_ignored(
    tmp_path: Path,
) -> None:
    source = _project_text().replace(
        "target static B 173.1 GPa",
        "target static B 173.1 GPa weight 0.5",
    )
    path = _write(tmp_path, source)

    with pytest.raises(InputFileError, match="elasticity target syntax"):
        parse_input_file(path)
