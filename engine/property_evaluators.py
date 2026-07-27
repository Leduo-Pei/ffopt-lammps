"""Composable property evaluators for one resolved force-field parameter set.

The evaluators intentionally call the validated LAMMPSRunner primitives. This
module owns dependency ordering and result composition; it does not duplicate
LAMMPS input generation, physical formulas, or objective definitions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
import inspect
import re
from threading import Lock
from typing import Any, Iterable

import numpy as np


KCAL_MOL_A2_TO_J_M2 = 0.69477


@dataclass(frozen=True)
class PropertyEvaluationContext:
    resolved: dict[str, float]
    eval_dir: str
    save_traj: bool = False
    seed_override: int | None = None


@dataclass
class PropertyStageResult:
    success: bool
    properties: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def failed(cls, error: str, properties: dict[str, float] | None = None):
        return cls(False, properties or {}, error)


class PropertyEvaluator:
    name = "base"
    dependencies: tuple[str, ...] = ()
    provides: frozenset[str] = frozenset()

    def evaluate(
        self,
        runner: Any,
        context: PropertyEvaluationContext,
        accumulated: dict[str, float],
    ) -> PropertyStageResult:
        raise NotImplementedError


class BulkEvaluator(PropertyEvaluator):
    name = "bulk"

    def evaluate(self, runner, context, accumulated) -> PropertyStageResult:
        properties = runner._run_bulk(
            context.resolved,
            context.eval_dir,
            context.save_traj,
            context.seed_override,
        )
        if properties is None:
            return PropertyStageResult.failed(
                "bulk LAMMPS failed or DATA_BULK line missing"
            )
        error = runner._bulk_sanity_check(properties)
        if error:
            return PropertyStageResult.failed(error, properties)
        return PropertyStageResult(True, properties)


class AdsorptionEvaluator(PropertyEvaluator):
    name = "adsorption"
    provides = frozenset({"ead", "E_complex", "E_slab", "E_mol"})

    def evaluate(self, runner, context, accumulated) -> PropertyStageResult:
        properties = runner._run_adsorption(
            context.resolved, context.eval_dir, context.save_traj
        )
        if properties is None:
            return PropertyStageResult.failed(
                "adsorption LAMMPS failed or DATA_AD line missing"
            )
        return PropertyStageResult(True, properties)


class SublimationEvaluator(PropertyEvaluator):
    name = "sublimation"
    dependencies = ("bulk",)
    provides = frozenset({
        "esub_proxy",
        "esub_proxy_kcal_mol",
        "esub_single_pe",
        "esub_bulk_pe_per_mol",
    })

    def evaluate(self, runner, context, accumulated) -> PropertyStageResult:
        bulk_pe = accumulated.get("pe")
        if bulk_pe is None:
            return PropertyStageResult.failed(
                "sublimation requires bulk property 'pe'"
            )
        import os

        result = runner._compute_sublimation_proxy_from_bulk_pe(
            context.resolved,
            os.path.join(context.eval_dir, "sublimation"),
            bulk_pe,
            context.save_traj,
        )
        if result is None:
            return PropertyStageResult.failed(
                "sublimation proxy LAMMPS failed or DATA_SUB line missing"
            )
        return PropertyStageResult(True, {
            "esub_proxy": result["esub_proxy_kj_mol"],
            "esub_proxy_kcal_mol": result["esub_proxy_kcal_mol"],
            "esub_single_pe": result["E_single_kcal"],
            "esub_bulk_pe_per_mol": result["E_bulk_per_molecule_kcal"],
        })


class SurfaceEvaluator(PropertyEvaluator):
    name = "surface"
    provides = frozenset({"surf_energy", "E_complete", "E_split", "A_xy"})

    def evaluate(self, runner, context, accumulated) -> PropertyStageResult:
        save_value = 1 if context.save_traj else 0
        variables = {
            "cutoff": runner.cutoff,
            "npt_seed": runner.surf_npt_seed,
            "equil_steps": runner.surf_equil,
            "prod_steps": runner.surf_prod,
            "save_traj": save_value,
            **(
                {"kspace_accuracy": runner.kspace_accuracy}
                if runner.use_charge else {}
            ),
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            complete_future = executor.submit(
                runner._run_surf,
                context.resolved,
                context.eval_dir,
                "complete",
                runner.surf_complete,
                variables,
            )
            split_future = executor.submit(
                runner._run_surf,
                context.resolved,
                context.eval_dir,
                "split",
                runner.surf_split,
                variables,
            )
            complete = complete_future.result()
            split = split_future.result()

        if complete is None:
            return PropertyStageResult.failed(
                "surf_complete LAMMPS failed or DATA_SURF line missing"
            )
        if split is None:
            return PropertyStageResult.failed(
                "surf_split LAMMPS failed or DATA_SURF line missing"
            )
        try:
            energy_complete = float(complete[runner.surf_E_key])
            energy_split = float(split[runner.surf_E_key])
            area = float(complete[runner.surf_A_key])
            if area == 0.0:
                raise ZeroDivisionError("surface area is zero")
            surface_energy = (
                (energy_split - energy_complete)
                / (2.0 * area)
                * KCAL_MOL_A2_TO_J_M2
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return PropertyStageResult.failed(
                f"surf_energy computation failed: {exc}"
            )

        properties = {
            "surf_energy": surface_energy,
            "E_complete": energy_complete,
            "E_split": energy_split,
            "A_xy": area,
        }
        if not np.isfinite(surface_energy):
            return PropertyStageResult.failed(
                f"non-finite surf_energy={surface_energy}", properties
            )
        if surface_energy <= runner.surf_energy_min:
            return PropertyStageResult.failed(
                f"surf_energy={surface_energy:.4f} J/m2 <= sanity min "
                f"{runner.surf_energy_min} J/m2",
                properties,
            )
        if surface_energy > runner.surf_energy_max:
            return PropertyStageResult.failed(
                f"surf_energy={surface_energy:.4f} J/m2 > sanity max "
                f"{runner.surf_energy_max} J/m2",
                properties,
            )
        return PropertyStageResult(True, properties)


DEFAULT_EVALUATORS = {
    evaluator.name: evaluator
    for evaluator in (
        BulkEvaluator(),
        AdsorptionEvaluator(),
        SublimationEvaluator(),
        SurfaceEvaluator(),
    )
}


class PropertyEvaluatorRegistry:
    """Registry for built-in and third-party property evaluators."""

    def __init__(self, evaluators: Iterable[PropertyEvaluator] = ()):
        self._evaluators: dict[str, PropertyEvaluator] = {}
        self._sources: dict[str, str] = {}
        self._discovered = False
        self._lock = Lock()
        self._discovery_lock = Lock()
        for evaluator in evaluators:
            self.register(evaluator, source="built-in")

    @staticmethod
    def _materialize(candidate: Any) -> PropertyEvaluator:
        if inspect.isclass(candidate) and issubclass(candidate, PropertyEvaluator):
            candidate = candidate()
        elif not isinstance(candidate, PropertyEvaluator) and callable(candidate):
            candidate = candidate()
        if not isinstance(candidate, PropertyEvaluator):
            raise TypeError(
                "Property evaluator plugins must expose a PropertyEvaluator "
                "instance, subclass, or zero-argument factory"
            )
        return candidate

    def register(
        self,
        evaluator: PropertyEvaluator | type[PropertyEvaluator] | Any,
        *,
        source: str = "runtime",
        replace: bool = False,
    ) -> PropertyEvaluator:
        instance = self._materialize(evaluator)
        name = str(instance.name).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(
                f"Invalid property evaluator name {name!r}; use snake_case"
            )
        with self._lock:
            if name in self._evaluators and not replace:
                raise ValueError(
                    f"Property evaluator {name!r} is already registered from "
                    f"{self._sources[name]}"
                )
            self._evaluators[name] = instance
            self._sources[name] = source
        return instance

    def discover(self) -> None:
        """Load ``ffopt.property_evaluators`` package entry points once."""
        with self._discovery_lock:
            if self._discovered:
                return
            discovered = metadata.entry_points()
            if hasattr(discovered, "select"):
                entries = discovered.select(group="ffopt.property_evaluators")
            else:  # pragma: no cover - Python/importlib-metadata compatibility
                entries = discovered.get("ffopt.property_evaluators", [])
            for entry in entries:
                try:
                    self.register(
                        entry.load(),
                        source=f"entry-point:{entry.value}",
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load property evaluator entry point "
                        f"{entry.name!r}: {exc}"
                    ) from exc
            self._discovered = True

    def evaluators(self, *, discover: bool = True) -> dict[str, PropertyEvaluator]:
        if discover:
            self.discover()
        return dict(self._evaluators)

    def describe(self, *, discover: bool = True) -> list[dict[str, Any]]:
        evaluators = self.evaluators(discover=discover)
        return [
            {
                "name": name,
                "source": self._sources[name],
                "dependencies": list(evaluator.dependencies),
                "provides": sorted(evaluator.provides),
                "class": f"{type(evaluator).__module__}.{type(evaluator).__qualname__}",
            }
            for name, evaluator in evaluators.items()
        ]


PROPERTY_EVALUATORS = PropertyEvaluatorRegistry(DEFAULT_EVALUATORS.values())


def register_property_evaluator(
    evaluator: PropertyEvaluator | type[PropertyEvaluator] | Any,
    *,
    replace: bool = False,
) -> PropertyEvaluator:
    """Register an evaluator from application code without an entry point."""
    return PROPERTY_EVALUATORS.register(evaluator, replace=replace)


class PropertyExecutionPlan:
    def __init__(self, evaluators: Iterable[PropertyEvaluator]):
        self.evaluators = tuple(evaluators)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.evaluators)

    def execute(
        self,
        runner: Any,
        context: PropertyEvaluationContext,
    ) -> PropertyStageResult:
        accumulated: dict[str, float] = {}
        for evaluator in self.evaluators:
            outcome = evaluator.evaluate(runner, context, accumulated)
            accumulated.update(outcome.properties)
            if not outcome.success:
                return PropertyStageResult.failed(outcome.error, accumulated)
        return PropertyStageResult(True, accumulated)


def _topological_order(
    names: set[str], evaluators: dict[str, PropertyEvaluator]
) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()

    unknown = names - evaluators.keys()
    if unknown:
        choices = ", ".join(sorted(evaluators))
        raise ValueError(
            f"Unknown property evaluator(s) {sorted(unknown)}. Available: {choices}"
        )

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise ValueError(f"Property evaluator dependency cycle at '{name}'")
        if name not in evaluators:
            raise ValueError(
                f"Property evaluator dependency {name!r} is not registered"
            )
        visiting.add(name)
        for dependency in evaluators[name].dependencies:
            names.add(dependency)
            visit(dependency)
        visiting.remove(name)
        ordered.append(name)

    for selected in evaluators:
        if selected in names:
            visit(selected)
    return ordered


def build_property_plan(config: dict[str, Any], runner: Any) -> PropertyExecutionPlan:
    available = PROPERTY_EVALUATORS.evaluators()
    selected = set()
    if runner.bulk_required:
        selected.add("bulk")
    if runner.ads_enabled:
        selected.add("adsorption")
    if runner.sub_enabled:
        selected.add("sublimation")
    if runner.compute_surface:
        selected.add("surface")

    for name, settings in config.get("property_evaluators", {}).items():
        enabled = settings if isinstance(settings, bool) else settings.get("enabled", True)
        if enabled:
            selected.add(name)
        else:
            selected.discard(name)

    ordered_names = _topological_order(selected, available)
    evaluators = [available[name] for name in ordered_names]

    provided = set()
    for evaluator in evaluators:
        provided.update(evaluator.provides)
        if evaluator.name == "bulk":
            provided.update(runner.bulk_col_map)
            provided.update(runner.bulk_targets)
            provided.update(runner.bulk_targets.values())
    active_targets = {
        name for name, info in config.get("targets", {}).items()
        if float(info.get("weight", 1.0)) > 0.0
    }
    missing = active_targets - provided
    if missing:
        raise ValueError(
            "No enabled property evaluator provides active target(s): "
            + ", ".join(sorted(missing))
        )
    return PropertyExecutionPlan(evaluators)
