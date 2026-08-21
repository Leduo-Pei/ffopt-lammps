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

0.0 2.8665 xlo xhi
0.0 2.8665 ylo yhi
0.0 2.8665 zlo zhi

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
    equilibration 20000
    production 40000
    seeds 101 202 303
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


def _project_text(elasticity: str | None = None) -> str:
    return f"""ffopt 1
project fe_elastic
material elemental Fe
crystal bcc
workflow bo validate

parameters
    range epsilon absolute 0.001 10
    range sigma absolute 0.001 5
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
        "fit_quality": {"minimum_r2": 0.98},
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
    assert dynamic["protocol"]["seeds"] == [101, 202, 303]
    assert dynamic["validation_protocol"]["seeds"] == [404, 505, 606]
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
        "method": "symmetric_energy_strain",
        "strain_magnitudes": [0.002, 0.004, 0.006],
        "replicate": [2, 2, 2],
        "temperature_k": 0.0,
    }

    baseline_hash = scientific_config_hash(_scientific_config(compiled.config))
    changed = source.replace("tier 20 percent", "tier 15 percent")
    changed_compiled = compile_input(parse_input_file(_write(tmp_path, changed)))
    changed_hash = scientific_config_hash(
        _scientific_config(changed_compiled.config)
    )
    assert changed_hash != baseline_hash


def test_legacy_dynamic_input_without_holdout_seeds_keeps_old_shape(
    tmp_path: Path,
) -> None:
    source = _project_text().replace(
        "    validation_seeds 404 505 606\n", ""
    )
    compiled = compile_input(parse_input_file(_write(tmp_path, source)))

    assert "validation_protocol" not in (
        compiled.config["elasticity"]["modules"]["dynamic"]
    )


@pytest.mark.parametrize("forbidden", ["K", "G", "E", "nu"])
def test_elasticity_rejects_derived_or_aliased_fit_targets(
    tmp_path: Path,
    forbidden: str,
) -> None:
    block = _elasticity_block(dynamic=False).replace(
        "target static B 173.1 GPa",
        f"target static {forbidden} 173.1 GPa",
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match="exactly B, Cprime, and C44") as exc:
        parse_input_file(path)
    assert exc.value.line == next(
        index
        for index, line in enumerate(path.read_text().splitlines(), 1)
        if f"target static {forbidden}" in line
    )


def test_dynamic_module_requires_its_own_complete_target_triplet(
    tmp_path: Path,
) -> None:
    block = _elasticity_block().replace(
        "    target dynamic C44 115.87 GPa\n", ""
    )
    path = _write(tmp_path, _project_text(block))

    with pytest.raises(InputFileError, match=r"dynamic module requires exactly.*C44"):
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
