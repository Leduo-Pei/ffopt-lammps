#!/usr/bin/env python3
"""Generate a resumable, multi-seed LAMMPS training dataset.

The design deliberately combines global Sobol coverage with local perturbations
around stable low-objective BO candidates.  The global half lets a surrogate
learn the parameter-to-property map; the local half resolves the useful basin.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config_loader import load_config
from .lammps_interface import LAMMPSRunner
from .parameter_space import build_parameter_space
from utils.objective_rescoring import objective_provenance


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def optional_float(value: object, default: float = np.nan) -> float:
    """Convert optional design metadata without treating blanks as failures."""
    if value is None or pd.isna(value):
        return default
    return float(value)


def optional_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def valid_source_rows(frame: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
    if "success" in frame:
        frame = frame.loc[
            frame["success"].astype(str).str.lower().eq("true")
        ].copy()
    needed = param_names + ["objective"]
    finite = np.isfinite(frame[needed].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[finite].sort_values("objective").reset_index(drop=True)


def generate_design(
    source: pd.DataFrame,
    param_space: list[tuple[str, float, float]],
    runner: LAMMPSRunner,
    n_points: int,
    elite_centers: int,
    radii: list[float],
    global_fraction: float,
    seed: int,
    center_selection: str = "top",
    center_pool_multiplier: int = 5,
) -> pd.DataFrame:
    names = [item[0] for item in param_space]
    lo = np.asarray([item[1] for item in param_space], dtype=float)
    hi = np.asarray([item[2] for item in param_space], dtype=float)
    span = hi - lo
    if source.empty:
        raise ValueError("The source BO table contains no successful finite rows.")
    if center_selection == "diverse":
        pool_count = min(
            len(source),
            max(elite_centers, elite_centers * center_pool_multiplier),
        )
        pool = source.head(pool_count).copy()
        pool_unit = (pool[names].to_numpy(dtype=float) - lo) / span
        selected = [0]
        nearest_sq = np.sum((pool_unit - pool_unit[0]) ** 2, axis=1)
        while len(selected) < min(elite_centers, len(pool)):
            nearest_sq[selected] = -np.inf
            next_index = int(np.argmax(nearest_sq))
            selected.append(next_index)
            distance_sq = np.sum(
                (pool_unit - pool_unit[next_index]) ** 2, axis=1
            )
            nearest_sq = np.minimum(nearest_sq, distance_sq)
        elite = pool.iloc[selected].copy().reset_index(drop=True)
    elif center_selection == "top":
        elite = source.head(min(elite_centers, len(source))).copy()
    else:
        raise ValueError("center_selection must be 'top' or 'diverse'")
    centers = (elite[names].to_numpy(dtype=float) - lo) / span

    if not 0.0 <= global_fraction <= 1.0:
        raise ValueError("global_fraction must lie between 0 and 1.")

    # A scrambled Sobol sequence gives deterministic, space-filling directions.
    pool_size = max(2048, int(2 ** np.ceil(np.log2(n_points * 4))))
    global_count = int(round(n_points * global_fraction))
    local_count = n_points - global_count
    global_unit = qmc.Sobol(d=len(names), scramble=True, seed=seed).random_base2(
        int(np.log2(pool_size))
    )
    local_unit = qmc.Sobol(d=len(names) + 1, scramble=True, seed=seed + 1).random_base2(
        int(np.log2(pool_size))
    )
    rng = np.random.default_rng(seed)
    radius_order = np.resize(np.asarray(radii, dtype=float), pool_size)
    rng.shuffle(radius_order)
    existing = {
        tuple(np.round(row, 12))
        for row in source[names].to_numpy(dtype=float)
    }
    accepted: list[dict] = []
    seen = set(existing)

    def add_candidate(values: np.ndarray, mode: str, center_index: int | None,
                      radius: float | None) -> bool:
        key = tuple(np.round(values, 12))
        if key in seen:
            return False
        params = dict(zip(names, values))
        if runner.feasibility_error(params) is not None:
            return False
        seen.add(key)
        record = {
            "candidate_id": len(accepted) + 1,
            "sampling_mode": mode,
            "center_rank": center_index + 1 if center_index is not None else 0,
            "center_objective": (
                float(elite.iloc[center_index]["objective"])
                if center_index is not None else np.nan
            ),
            "radius": radius,
        }
        record.update(params)
        accepted.append(record)
        return True

    if global_count > 0:
        for row in global_unit:
            add_candidate(lo + row * span, "global_sobol", None, None)
            if len(accepted) >= global_count:
                break

    local_added = 0
    if local_count > 0:
        for row_index, row in enumerate(local_unit):
            center_index = min(int(row[0] * len(centers)), len(centers) - 1)
            direction = 2.0 * row[1:] - 1.0
            radius = float(radius_order[row_index])
            normalized = np.clip(
                centers[center_index] + radius * direction, 0.0, 1.0
            )
            if add_candidate(values=lo + normalized * span, mode="local_elite",
                             center_index=center_index, radius=radius):
                local_added += 1
            if local_added >= local_count:
                break

    if len(accepted) < n_points:
        raise RuntimeError(
            f"Only generated {len(accepted)}/{n_points} feasible unique points "
            f"({global_count} global requested, {local_count} local requested). "
            "Increase the Sobol pool or relax the sampling radii."
        )
    return pd.DataFrame(accepted)


def evaluate_candidate(
    runner: LAMMPSRunner,
    row: dict,
    param_names: list[str],
    target_names: list[str],
    seeds: list[int],
    work_root: Path,
    keep_work: bool,
) -> tuple[dict, list[dict]]:
    candidate_id = int(row["candidate_id"])
    params = {name: float(row[name]) for name in param_names}
    candidate_dir = work_root / f"candidate_{candidate_id:05d}"
    results = runner.evaluate_replicates(
        params, str(candidate_dir), seeds, save_traj=False,
        reuse_existing=True,
    )
    replicate_rows = []
    successful = []
    for seed, result in zip(seeds, results):
        rep = {
            "candidate_id": candidate_id,
            "seed": int(seed),
            "success": bool(result.success),
            "objective": float(result.objective),
            "error_msg": result.error_msg,
        }
        rep.update(objective_provenance(runner.targets))
        rep.update(params)
        for prop in target_names:
            rep[f"calc_{prop}"] = result.properties.get(prop, np.nan)
            rep[f"error_{prop}"] = result.per_property_error.get(prop, np.nan)
        replicate_rows.append(rep)
        if result.success and np.isfinite(result.objective):
            successful.append(result)

    summary = {
        "candidate_id": candidate_id,
        "center_rank": optional_int(row.get("center_rank")),
        "center_objective": optional_float(row.get("center_objective")),
        "radius": optional_float(row.get("radius")),
        "sampling_mode": str(row.get("sampling_mode", "stability_audit")),
        "success": len(successful) == len(seeds),
        "n_success": len(successful),
        "n_seeds": len(seeds),
        "failure_rate": 1.0 - len(successful) / len(seeds),
        "error_msg": " | ".join(
            result.error_msg for result in results if not result.success
        ),
    }
    summary.update(objective_provenance(runner.targets))
    summary.update(params)
    if successful:
        objectives = np.asarray([result.objective for result in successful])
        summary["objective_mean"] = float(objectives.mean())
        summary["objective_std"] = float(objectives.std(ddof=0))
        summary["objective"] = summary["objective_mean"] + summary["objective_std"]
        for prop in target_names:
            values = np.asarray([
                result.properties.get(prop, np.nan) for result in successful
            ], dtype=float)
            summary[f"calc_{prop}"] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            )
            summary[f"std_{prop}"] = (
                float(np.nanstd(values)) if np.isfinite(values).any() else np.nan
            )
    else:
        summary["objective_mean"] = np.nan
        summary["objective_std"] = np.nan
        summary["objective"] = np.nan
        for prop in target_names:
            summary[f"calc_{prop}"] = np.nan
            summary[f"std_{prop}"] = np.nan

    if not keep_work and summary["success"]:
        shutil.rmtree(candidate_dir, ignore_errors=True)
    return summary, replicate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--elite-centers", type=int, default=48)
    parser.add_argument(
        "--center-selection", choices=["top", "diverse"], default="top",
        help="Use the best rows directly or select dispersed centers from a "
             "larger low-objective pool.",
    )
    parser.add_argument("--center-pool-multiplier", type=int, default=5)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=[0.025, 0.05, 0.10, 0.20]
    )
    parser.add_argument(
        "--global-fraction", type=float, default=0.50,
        help="Fraction of points sampled across the full feasible parameter box."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--design-seed", type=int, default=20260709)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--design-only", action="store_true",
        help="Generate or validate design.csv and metadata.json, then exit."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("adsorption", {})["enabled"] = False
    runner = LAMMPSRunner(config)
    param_space = build_parameter_space(config)
    param_names = [item[0] for item in param_space]
    target_names = [
        name for name, info in config["targets"].items()
        if float(info.get("weight", 1.0)) > 0.0
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output_dir / "work"
    work_root.mkdir(exist_ok=True)
    design_path = args.output_dir / "design.csv"
    result_path = args.output_dir / "local_results.csv"
    replicate_path = args.output_dir / "local_replicates.csv"
    metadata_path = args.output_dir / "metadata.json"

    source = valid_source_rows(pd.read_csv(args.source), param_names)
    if design_path.exists():
        design = pd.read_csv(design_path)
        if len(design) != args.n_points:
            raise ValueError(
                f"Existing design has {len(design)} points, requested {args.n_points}."
            )
        print(f"Loaded existing deterministic design: {design_path}")
    else:
        design = generate_design(
            source, param_space, runner, args.n_points, args.elite_centers,
            args.radii, args.global_fraction, args.design_seed,
            args.center_selection, args.center_pool_multiplier,
        )
        atomic_csv(design, design_path)
        metadata_path.write_text(json.dumps({
            "config": args.config,
            "source": str(args.source),
            "n_points": args.n_points,
            "elite_centers": args.elite_centers,
            "center_selection": args.center_selection,
            "center_pool_multiplier": args.center_pool_multiplier,
            "radii": args.radii,
            "global_fraction": args.global_fraction,
            "seeds": args.seeds,
            "design_seed": args.design_seed,
            "parameter_names": param_names,
            "target_names": target_names,
        }, indent=2), encoding="utf-8")
        print(f"Generated deterministic design: {design_path}")

    if args.design_only:
        print(
            f"Design-only complete: {len(design)} feasible points -> "
            f"{design_path}"
        )
        return

    summaries = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    replicates = (
        pd.read_csv(replicate_path).to_dict("records")
        if replicate_path.exists() else []
    )
    requested_seeds = {int(seed) for seed in args.seeds}
    completed: set[int] = set()
    if replicates:
        replicate_frame = pd.DataFrame(replicates)
        for candidate_id, group in replicate_frame.groupby("candidate_id"):
            observed = set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int))
            if requested_seeds.issubset(observed):
                completed.add(int(candidate_id))
    else:
        completed = {
            int(row["candidate_id"]) for row in summaries
            if int(row.get("n_seeds", 0)) >= len(requested_seeds)
        }
    pending = [
        row for row in design.to_dict("records")
        if int(row["candidate_id"]) not in completed
    ]
    workers = args.max_workers or int(config["parallel"]["max_workers"])
    workers = max(1, min(workers, len(pending))) if pending else 0
    print(
        f"Local sampling: {len(design)} points x {len(args.seeds)} seeds | "
        f"completed={len(completed)} pending={len(pending)} workers={workers}"
    )
    if not pending:
        print("Dataset is already complete.")
        return

    started = time.time()
    task_exceptions: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                evaluate_candidate, runner, row, param_names, target_names,
                args.seeds, work_root, args.keep_work,
            ): int(row["candidate_id"])
            for row in pending
        }
        for future in as_completed(future_map):
            candidate_id = future_map[future]
            try:
                summary, rep_rows = future.result()
            except Exception as exc:
                print(f"  candidate {candidate_id:05d} exception: {exc}", flush=True)
                task_exceptions.append((candidate_id, str(exc)))
                continue
            replicates = [
                row for row in replicates
                if int(row["candidate_id"]) != candidate_id
            ]
            replicates.extend(rep_rows)
            atomic_csv(pd.DataFrame(replicates), replicate_path)
            summaries = [
                row for row in summaries
                if int(row["candidate_id"]) != candidate_id
            ]
            summaries.append(summary)
            summaries.sort(key=lambda item: int(item["candidate_id"]))
            atomic_csv(pd.DataFrame(summaries), result_path)
            done = len(summaries)
            if done <= 10 or done % 25 == 0 or done == len(design):
                print(
                    f"  completed {done}/{len(design)} | "
                    f"candidate={candidate_id:05d} "
                    f"objective={summary['objective']:.6f} "
                    f"success={summary['success']} "
                    f"elapsed={(time.time() - started) / 60:.1f} min",
                    flush=True,
                )

    if task_exceptions:
        examples = "; ".join(
            f"{candidate_id:05d}: {message}"
            for candidate_id, message in task_exceptions[:5]
        )
        raise RuntimeError(
            f"{len(task_exceptions)} candidate task(s) raised exceptions. "
            f"Examples: {examples}"
        )


if __name__ == "__main__":
    main()
