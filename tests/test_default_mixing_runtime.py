from __future__ import annotations

import numpy as np
import pytest

from engine.lammps_interface import LAMMPSRunner
from engine.nn_surrogate import PhysicsFeatureBuilder
from workflow.cli import _mixing_rule_description


@pytest.mark.parametrize(
    ("rule", "description"),
    [
        ("default", "epsilon geometric, sigma geometric"),
        ("geometric", "epsilon geometric, sigma geometric"),
        ("arithmetic", "epsilon geometric, sigma arithmetic"),
        ("none", "explicit unlike-pair coefficients"),
    ],
)
def test_explain_reports_effective_lammps_mixing(rule, description) -> None:
    assert _mixing_rule_description(rule) == description


def _writer(rule: str) -> LAMMPSRunner:
    runner = LAMMPSRunner.__new__(LAMMPSRunner)
    runner.mixing_rule = rule
    runner.explicit_pairs_cfg = []
    runner.atom_types = [
        {"type": 1, "label": "A"},
        {"type": 2, "label": "B"},
    ]
    runner.use_charge = False
    runner.metal_label = "Au"
    return runner


def _resolved():
    return {
        "A_epsilon": 1.0,
        "A_sigma": 2.0,
        "B_epsilon": 4.0,
        "B_sigma": 3.0,
    }


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("default", None),
        ("none", None),
        ("geometric", "pair_modify mix geometric"),
        ("arithmetic", "pair_modify mix arithmetic"),
    ],
)
def test_bulk_pair_writer_delegates_default_but_keeps_explicit_rules(
    tmp_path, rule, expected
):
    path = _writer(rule)._build_pair_coeffs(_resolved(), str(tmp_path))
    text = open(path, encoding="utf-8").read()

    assert "pair_modify mix default" not in text
    if expected is None:
        assert "pair_modify mix" not in text
    else:
        assert expected in text
    assert "pair_coeff 1 1 1.0000000000 2.0000000000" in text
    assert "pair_coeff 2 2 4.0000000000 3.0000000000" in text


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("default", None),
        ("none", None),
        ("geometric", "pair_modify mix geometric"),
        ("arithmetic", "pair_modify mix arithmetic"),
    ],
)
def test_adsorption_pair_writer_matches_bulk_mixing_semantics(
    tmp_path, rule, expected
):
    runner = _writer(rule)
    path = runner._write_pair_coeffs_system(
        _resolved(), str(tmp_path), {1: "Au", 2: "B"}, has_charge=False
    )
    text = open(path, encoding="utf-8").read()

    assert "pair_modify mix default" not in text
    if expected is None:
        assert "pair_modify mix" not in text
    else:
        assert expected in text
    assert "pair_coeff 1 1" not in text
    assert "pair_coeff 2 2 4.0000000000 3.0000000000" in text


def _feature_config(rule: str = "default", pair_style: str | None = "lj/cut"):
    config = {
        "charge": {"enabled": False},
        "pair_params": {"mixing_rule": rule},
        "atom_types": [
            {"type": 1, "label": "A", "params": {}},
            {"type": 2, "label": "B", "params": {}},
        ],
    }
    if pair_style is not None:
        config["lammps"] = {"pair_style": pair_style}
    return config


@pytest.mark.parametrize(
    "pair_style", ["lj/cut", "lj/cut/coul/long", "lj/cut/coul/cut"]
)
def test_default_nn_features_use_lammps_geometric_rule_without_new_parameters(
    pair_style,
):
    param_names = ["A_epsilon", "A_sigma", "B_epsilon", "B_sigma"]
    default_builder = PhysicsFeatureBuilder(
        param_names, _feature_config(pair_style=pair_style)
    )
    geometric_builder = PhysicsFeatureBuilder(
        param_names, _feature_config(rule="geometric", pair_style=pair_style)
    )
    raw = np.array([[1.0, 4.0, 9.0, 1.0]])

    assert default_builder.mixing_rule == "geometric"
    assert default_builder.param_names == param_names
    assert default_builder.extra_feature_names == geometric_builder.extra_feature_names
    np.testing.assert_allclose(
        default_builder.transform(raw), geometric_builder.transform(raw)
    )
    # Raw optimization coordinates remain the first four columns; the two
    # cross quantities are derived NN features, not new search dimensions.
    np.testing.assert_allclose(default_builder.transform(raw)[0, :4], raw[0])
    assert default_builder.transform(raw)[0, 4] == pytest.approx(3.0)
    assert default_builder.transform(raw)[0, 5] == pytest.approx(2.0)


def test_default_nn_features_can_read_pair_style_from_lammps_input(tmp_path):
    input_path = tmp_path / "in.bulk"
    input_path.write_text(
        "# pair_style eam must be ignored because it is commented\n"
        "pair_style lj/cut/coul/long ${cutoff}\n",
        encoding="utf-8",
    )
    config = _feature_config(pair_style=None)
    config["lammps"] = {"bulk_input": str(input_path)}

    builder = PhysicsFeatureBuilder(
        ["A_epsilon", "A_sigma", "B_epsilon", "B_sigma"], config
    )

    assert builder.mixing_rule == "geometric"


@pytest.mark.parametrize("pair_style", ["eam", "hybrid", "buck/coul/long"])
def test_default_nn_features_fail_closed_for_unknown_pair_style(pair_style):
    with pytest.raises(ValueError, match="cannot resolve LAMMPS default mixing"):
        PhysicsFeatureBuilder(
            ["A_epsilon", "A_sigma", "B_epsilon", "B_sigma"],
            _feature_config(pair_style=pair_style),
        )


def test_default_nn_features_fail_closed_without_pair_style():
    with pytest.raises(ValueError, match="no pair_style"):
        PhysicsFeatureBuilder(
            ["A_epsilon", "A_sigma", "B_epsilon", "B_sigma"],
            _feature_config(pair_style=None),
        )


def test_persisted_effective_rule_reloads_without_original_input_file():
    config = _feature_config(pair_style=None)
    config["physics_feature_mixing_rule"] = "geometric"

    builder = PhysicsFeatureBuilder(
        ["A_epsilon", "A_sigma", "B_epsilon", "B_sigma"], config
    )

    assert builder.mixing_rule == "geometric"
