"""Composable property evaluators for one resolved force-field parameter set.

The evaluators intentionally call the validated LAMMPSRunner primitives. This
module owns dependency ordering and result composition; it does not duplicate
LAMMPS input generation, physical formulas, or objective definitions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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


def _topological_order(names: set[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()

    unknown = names - DEFAULT_EVALUATORS.keys()
    if unknown:
        choices = ", ".join(sorted(DEFAULT_EVALUATORS))
        raise ValueError(
            f"Unknown property evaluator(s) {sorted(unknown)}. Available: {choices}"
        )

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise ValueError(f"Property evaluator dependency cycle at '{name}'")
        visiting.add(name)
        for dependency in DEFAULT_EVALUATORS[name].dependencies:
            names.add(dependency)
            visit(dependency)
        visiting.remove(name)
        ordered.append(name)

    for selected in DEFAULT_EVALUATORS:
        if selected in names:
            visit(selected)
    return ordered


def build_property_plan(config: dict[str, Any], runner: Any) -> PropertyExecutionPlan:
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

    ordered_names = _topological_order(selected)
    evaluators = [DEFAULT_EVALUATORS[name] for name in ordered_names]

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
