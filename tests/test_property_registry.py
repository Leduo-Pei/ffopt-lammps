from engine import property_evaluators as module
from ffopt.properties import PropertyEvaluator as PublicPropertyEvaluator
from engine.property_evaluators import (
    DEFAULT_EVALUATORS,
    PropertyEvaluator,
    PropertyEvaluatorRegistry,
    PropertyStageResult,
    build_property_plan,
)


class DemoEvaluator(PropertyEvaluator):
    name = "demo_property"
    provides = frozenset({"demo_value"})

    def evaluate(self, runner, context, accumulated):
        return PropertyStageResult(True, {"demo_value": 1.0})


def test_public_plugin_api_reexports_evaluator_base():
    assert PublicPropertyEvaluator is PropertyEvaluator


def test_registry_accepts_public_evaluator_class():
    registry = PropertyEvaluatorRegistry()
    evaluator = registry.register(DemoEvaluator, source="test")
    assert evaluator.name == "demo_property"
    description = registry.describe(discover=False)[0]
    assert description["source"] == "test"
    assert description["provides"] == ["demo_value"]


def test_build_plan_can_use_registered_plugin(monkeypatch):
    registry = PropertyEvaluatorRegistry(DEFAULT_EVALUATORS.values())
    registry.register(DemoEvaluator, source="test")
    monkeypatch.setattr(module, "PROPERTY_EVALUATORS", registry)

    runner = type("Runner", (), {
        "bulk_required": False,
        "ads_enabled": False,
        "sub_enabled": False,
        "compute_surface": False,
        "bulk_col_map": [],
        "bulk_targets": {},
    })()
    plan = build_property_plan({
        "property_evaluators": {"demo_property": {"enabled": True}},
        "targets": {"demo_value": {"value": 1.0, "weight": 1.0}},
    }, runner)
    assert plan.names == ("demo_property",)


def test_registry_discovers_entry_point(monkeypatch):
    class FakeEntryPoint:
        name = "demo_property"
        value = "demo_package:DemoEvaluator"

        @staticmethod
        def load():
            return DemoEvaluator

    class FakeEntryPoints(list):
        def select(self, *, group):
            assert group == "ffopt.property_evaluators"
            return self

    monkeypatch.setattr(
        module.metadata, "entry_points", lambda: FakeEntryPoints([FakeEntryPoint()])
    )
    registry = PropertyEvaluatorRegistry()
    assert "demo_property" in registry.evaluators()
    assert registry.describe()[0]["source"].startswith("entry-point:")


def test_plugin_missing_dependency_has_actionable_error(monkeypatch):
    class MissingDependencyEvaluator(DemoEvaluator):
        name = "dependent_demo"
        dependencies = ("not_installed",)

    registry = PropertyEvaluatorRegistry(DEFAULT_EVALUATORS.values())
    registry.register(MissingDependencyEvaluator, source="test")
    monkeypatch.setattr(module, "PROPERTY_EVALUATORS", registry)
    runner = type("Runner", (), {
        "bulk_required": False,
        "ads_enabled": False,
        "sub_enabled": False,
        "compute_surface": False,
        "bulk_col_map": [],
        "bulk_targets": {},
    })()
    try:
        build_property_plan({
            "property_evaluators": {"dependent_demo": True},
            "targets": {"demo_value": {"value": 1.0}},
        }, runner)
    except ValueError as exc:
        assert "not_installed" in str(exc)
        assert "not registered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing plugin dependency unexpectedly passed")
