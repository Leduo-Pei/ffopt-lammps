#!/usr/bin/env python3
"""Resumable multi-seed stability audit for top force-field candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import load_config
from .lammps_interface import LAMMPSRunner
from .local_sampling import atomic_csv, evaluate_candidate, valid_source_rows
from .optimizer import ForceFieldOptimizer


def unique_top(frame: pd.DataFrame, names: list[str], top_k: int) -> pd.DataFrame:
    """Return the best distinct parameter vectors with audit metadata."""
    frame = frame.sort_values("objective").copy()
    keys = frame[names].round(12).astype(str).agg("|".join, axis=1)
    frame = frame.loc[~keys.duplicated()].head(top_k).reset_index(drop=True)
    if "candidate_id" in frame.columns:
        if "source_candidate_id" in frame.columns:
            frame = frame.drop(columns=["source_candidate_id"])
        frame = frame.rename(columns={"candidate_id": "source_candidate_id"})
    frame.insert(0, "candidate_id", np.arange(1, len(frame) + 1))
    frame["center_rank"] = frame["candidate_id"]
    frame["center_objective"] = frame["objective"]
    frame["radius"] = 0.0
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--label", default=None,
        help="Optional source label to audit, for example al_round_1."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--design-only", action="store_true",
        help="Write or validate design.csv without running LAMMPS."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("adsorption", {})["enabled"] = False
    runner = LAMMPSRunner(config)
    space = ForceFieldOptimizer._build_param_space(config)
    names = [item[0] for item in space]
    targets = [
        name for name, info in config["targets"].items()
        if float(info.get("weight", 1.0)) > 0.0
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output_dir / "work"
    work_root.mkdir(exist_ok=True)
    design_path = args.output_dir / "design.csv"
    result_path = args.output_dir / "stable_results.csv"
    replicate_path = args.output_dir / "stability_replicates.csv"

    if design_path.exists():
        design = pd.read_csv(design_path)
        if len(design) != args.top_k:
            raise ValueError(
                f"Existing design has {len(design)} rows, requested {args.top_k}."
            )
    else:
        source_frame = pd.read_csv(args.source)
        if args.label is not None:
            if "label" not in source_frame.columns:
                raise ValueError(
                    f"--label {args.label!r} requested but source has no label column"
                )
            source_frame = source_frame.loc[
                source_frame["label"].astype(str).eq(args.label)
            ].copy()
            if source_frame.empty:
                raise ValueError(
                    f"No source rows matched label {args.label!r}"
                )
            # active_learning.evaluate_batch preserves this order, so the
            # index maps directly to lammps_round_N/eval_NNNN for seed reuse.
            source_frame["source_eval_index"] = np.arange(len(source_frame))
        source = valid_source_rows(source_frame, names)
        design = unique_top(source, names, args.top_k)
        atomic_csv(design, design_path)

    if args.design_only:
        print(f"Design-only complete: {len(design)} candidates -> {design_path}")
        return

    completed: set[int] = set()
    summaries: list[dict] = []
    replicates: list[dict] = []
    if result_path.exists():
        old = pd.read_csv(result_path)
        summaries = old.to_dict("records")
        completed = set(old.loc[old["n_seeds"] == len(args.seeds), "candidate_id"])
    if replicate_path.exists():
        replicates = pd.read_csv(replicate_path).to_dict("records")

    pending = [
        row for row in design.to_dict("records")
        if int(row["candidate_id"]) not in completed
    ]
    workers = args.max_workers or int(config["parallel"]["max_workers"])
    workers = max(1, min(workers, len(pending))) if pending else 0
    print(
        f"Stability audit: top {len(design)} distinct candidates x "
        f"{len(args.seeds)} seeds | completed={len(completed)} "
        f"pending={len(pending)} workers={workers}"
    )

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    evaluate_candidate, runner, row, names, targets,
                    args.seeds, work_root, args.keep_work
                ): int(row["candidate_id"])
                for row in pending
            }
            for future in as_completed(futures):
                candidate_id = futures[future]
                summary, new_replicates = future.result()
                summaries = [
                    row for row in summaries
                    if int(row["candidate_id"]) != candidate_id
                ]
                replicates = [
                    row for row in replicates
                    if int(row["candidate_id"]) != candidate_id
                ]
                summaries.append(summary)
                replicates.extend(new_replicates)
                summary_frame = pd.DataFrame(summaries).sort_values("objective")
                replicate_frame = pd.DataFrame(replicates).sort_values(
                    ["candidate_id", "seed"]
                )
                atomic_csv(summary_frame, result_path)
                atomic_csv(replicate_frame, replicate_path)
                print(
                    f"  candidate {candidate_id:02d}: "
                    f"mean={summary['objective_mean']:.6f} "
                    f"std={summary['objective_std']:.6f} "
                    f"robust={summary['objective']:.6f}"
                )

    final = pd.DataFrame(summaries).sort_values("objective")
    atomic_csv(final, result_path)
    print("\nRobust ranking (mean + std):")
    print(final[[
        "candidate_id", "center_objective", "objective_mean",
        "objective_std", "objective", "n_success", "n_seeds"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
