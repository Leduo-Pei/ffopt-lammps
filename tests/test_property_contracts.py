from __future__ import annotations

import pytest

from engine.property_contracts import (
    DEFAULT_PROPERTY_COST_CLASS,
    DEFAULT_PROPERTY_FIDELITY,
    DEFAULT_PROPERTY_ROLES,
    PropertyContract,
    PropertyCostClass,
    PropertyRole,
)
from engine.property_evaluators import (
    PropertyEvaluator,
    PropertyEvaluatorRegistry,
    PropertyStageResult,
)
from ffopt.properties import PropertyContract as PublicPropertyContract
from ffopt.properties import PropertyRole as PublicPropertyRole


class LegacyEvaluator(PropertyEvaluator):
    name = "legacy_value"
    provides = frozenset({"legacy_output"})

    def evaluate(self, runner, context, accumulated):
        return PropertyStageResult(True, {"legacy_output": 1.0})


class StructuralPrerequisite(PropertyEvaluator):
    name = "structural_prerequisite"
    roles = frozenset({"constraint", "validation"})
    fidelity = "structural"
    cost_class = "low"
    provides = frozenset({"volume"})

    def evaluate(self, runner, context, accumulated):
        return PropertyStageResult(True, {"volume": 10.0})


class StaticElasticity(PropertyEvaluator):
    name = "static_elasticity"
    contract = PropertyContract(
        roles=frozenset({PropertyRole.OBJECTIVE, PropertyRole.PROMOTION}),
        fidelity="static_0k",
        cost_class=PropertyCostClass.HIGH,
        dependencies=("structural_prerequisite",),
        provides=frozenset({"K_0k", "Cprime_0k", "C44_0k"}),
    )

    def evaluate(self, runner, context, accumulated):
        assert accumulated["volume"] == 10.0
        return PropertyStageResult(True, {"K_0k": 173.0})


def test_contract_api_is_public() -> None:
    assert PublicPropertyContract is PropertyContract
    assert PublicPropertyRole is PropertyRole


def test_legacy_evaluator_gets_compatible_contract_defaults() -> None:
    registry = PropertyEvaluatorRegistry([LegacyEvaluator()])
    contract = registry.contract("legacy_value", discover=False)
    assert contract.roles == DEFAULT_PROPERTY_ROLES
    assert contract.fidelity == DEFAULT_PROPERTY_FIDELITY
    assert contract.cost_class == DEFAULT_PROPERTY_COST_CLASS
    assert contract.dependencies == ()
    assert contract.provides == frozenset({"legacy_output"})

    description = registry.describe(discover=False)[0]
    assert description["roles"] == ["objective", "validation"]
    assert description["fidelity"] == "unspecified"
    assert description["cost_class"] == "medium"
    assert description["provides"] == ["legacy_output"]


def test_class_attribute_contract_declaration_is_normalized() -> None:
    registry = PropertyEvaluatorRegistry([StructuralPrerequisite()])
    contract = registry.contract("structural_prerequisite", discover=False)
    assert contract.roles == frozenset({
        PropertyRole.CONSTRAINT,
        PropertyRole.VALIDATION,
    })
    assert contract.fidelity == "structural"
    assert contract.cost_class is PropertyCostClass.LOW


def test_explicit_contract_is_described_as_json_compatible_metadata() -> None:
    registry = PropertyEvaluatorRegistry([
        StructuralPrerequisite(),
        StaticElasticity(),
    ])
    description = {
        item["name"]: item for item in registry.describe(discover=False)
    }["static_elasticity"]
    assert description["roles"] == ["objective", "promotion"]
    assert description["fidelity"] == "static_0k"
    assert description["cost_class"] == "high"
    assert description["dependencies"] == ["structural_prerequisite"]
    assert description["provides"] == ["C44_0k", "Cprime_0k", "K_0k"]


def test_registry_filters_by_role_fidelity_and_cost() -> None:
    registry = PropertyEvaluatorRegistry([
        LegacyEvaluator(),
        StructuralPrerequisite(),
        StaticElasticity(),
    ])
    assert tuple(registry.select(
        role="constraint", discover=False
    )) == ("structural_prerequisite",)
    assert tuple(registry.select(
        fidelity="static_0k", discover=False
    )) == ("static_elasticity",)
    assert tuple(registry.select(
        role=PropertyRole.PROMOTION,
        fidelity="static_0k",
        cost_class="high",
        discover=False,
    )) == ("static_elasticity",)


def test_filtered_plan_adds_nonmatching_dependencies_topologically() -> None:
    registry = PropertyEvaluatorRegistry([
        StaticElasticity(),
        StructuralPrerequisite(),
    ])
    plan = registry.build_plan(
        role="promotion", fidelity="static_0k", discover=False
    )
    assert plan.names == ("structural_prerequisite", "static_elasticity")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"roles": ()}, "must not be empty"),
        ({"roles": ("objective", "objective")}, "duplicates"),
        ({"roles": ("fit",)}, "Invalid property role"),
        ({"fidelity": "static 0K"}, "Invalid property fidelity"),
        ({"cost_class": "slow"}, "Invalid property cost class"),
        ({"dependencies": ("Not-Snake",)}, "Invalid dependency"),
        ({"dependencies": ("bulk", "bulk")}, "duplicates"),
        ({"provides": ("surface-energy",)}, "Invalid provided property"),
        ({"provides": ("K", "K")}, "duplicates"),
    ],
)
def test_contract_rejects_ambiguous_or_invalid_metadata(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        PropertyContract(**kwargs)


def test_registry_rejects_self_dependency() -> None:
    class RecursiveEvaluator(PropertyEvaluator):
        name = "recursive"
        dependencies = ("recursive",)

    with pytest.raises(ValueError, match="cannot depend on itself"):
        PropertyEvaluatorRegistry([RecursiveEvaluator()])


def test_registry_rejects_whitespace_in_evaluator_name() -> None:
    class PaddedNameEvaluator(PropertyEvaluator):
        name = " padded_name"

    with pytest.raises(ValueError, match="Invalid property evaluator name"):
        PropertyEvaluatorRegistry([PaddedNameEvaluator()])


def test_registry_rejects_conflicting_explicit_and_legacy_metadata() -> None:
    class ConflictingEvaluator(PropertyEvaluator):
        name = "conflicting"
        contract = PropertyContract(
            dependencies=("first",),
            provides=frozenset({"value"}),
        )
        dependencies = ("second",)

    with pytest.raises(ValueError, match="conflicting 'dependencies'"):
        PropertyEvaluatorRegistry([ConflictingEvaluator()])


def test_plan_rejects_mixed_explicit_names_and_contract_filters() -> None:
    registry = PropertyEvaluatorRegistry([LegacyEvaluator()])
    with pytest.raises(ValueError, match="either explicit names or contract filters"):
        registry.build_plan(
            names=("legacy_value",), role="objective", discover=False
        )


def test_contract_filter_values_are_validated_even_when_registry_is_empty() -> None:
    registry = PropertyEvaluatorRegistry()
    with pytest.raises(ValueError, match="Invalid property role"):
        registry.select(role="fit", discover=False)
    with pytest.raises(ValueError, match="Invalid property fidelity"):
        registry.build_plan(fidelity="300 K", discover=False)
