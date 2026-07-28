#!/usr/bin/env python3
"""
active_learning.py  |  Active Learning Refinement Loop  |  v8

Workflow per round:
  1. Sample candidate pool (Sobol/LHS/random, size=n_candidate_pool)
  2. Rank by aggregated uncertainty: sum_prop  std_prop(x) / |target_prop|
     (normalized sum across all per-property EnsembleMLPs)
  3. Select top-N by highest uncertainty -> LAMMPS evaluation
  4. Append new evals -> retrain NNSurrogate -> reload ensembles
  5. Early-stop when max aggregated uncertainty < uncertainty_threshold

Usage:
  python -m engine.active_learning --config runs/.../_configs/project_local.json \\
      --bo-dir runs/.../bo --nn-dir runs/.../nn

Outputs (active_learning_output/):
  data_round_{N}/all_results.csv   - accumulated data after round N
  nn_round_{N}/                    - NNSurrogate outputs after round N
  nn_round_final/                  - copy of last nn_round_* (run.py target)
  active_learning_history.json     - per-round summary for viz
  final_parameters.json            - best LAMMPS-verified accumulated point
"""

import argparse
import glob as _glob
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor
from .config_loader import load_config as load_expanded_config, save_config_snapshot

# Same directory as active_learning.py -> lammps_interface, optimizer, nn_surrogate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .lammps_interface import LAMMPSRunner
from .nn_surrogate import (EnsembleMLP, NNSurrogate, PhysicsFeatureBuilder,
                           load_ensemble_from_file)

# Type alias for per-property ensemble dict returned by load_ensemble_from_file
from typing import Dict as _Dict
from .optimizer import ForceFieldOptimizer
from utils.objective_rescoring import (
    active_targets,
    objective_provenance,
    validate_objective_provenance,
)


# ============================================================================
# ActiveLearner
# ============================================================================

class ActiveLearner:
    """
    Iterative active-learning loop for force field parameter refinement.

    Reads
    -----
    config            : parsed v8 config.yaml
    bo_dir            : path to BO output directory (stable_results.csv preferred,
                        all_results.csv fallback)
    nn_dir            : path to initial NNSurrogate output (contains forward_nn.pt)

    Writes
    ------
    active_learning_output/   (or --output-dir)
      data_round_{r}/all_results.csv     - accumulated training data after round r
      nn_round_{r}/                      - NNSurrogate outputs after round r
      nn_round_final/                    - copy of last nn_round_*/
      active_learning_history.json       - per-round summary consumed by viz
    """

    def __init__(self,
                 config:       dict,
                 bo_dir:       str,
                 nn_dir:       str,
                 output_dir:   str,
                 no_validate:  bool = False,
                 save_traj:    bool = False,
                 resume:       bool = False):
        self.config      = config
        self.bo_dir      = bo_dir
        self.nn_dir      = nn_dir
        self.output_dir  = output_dir
        self.no_validate = no_validate
        self.save_traj   = save_traj
        self.resume      = resume

        # -- active_learning config --------------------------------------------
        al_cfg = config["active_learning"]

        self.n_rounds               = al_cfg["n_rounds"]               # int
        self.n_candidates_per_round = al_cfg["n_candidates_per_round"] # int
        self.uncertainty_threshold  = al_cfg["uncertainty_threshold"]  # float
        self.candidate_sampling     = al_cfg["candidate_sampling"]     # sobol|lhs|random
        self.n_candidate_pool       = al_cfg["n_candidate_pool"]       # int
        self.objective_bias         = float(al_cfg.get("objective_bias", 0.25))
        self.sampling_domain        = al_cfg.get("sampling_domain", "global")
        self.envelope_margin        = float(al_cfg.get("envelope_margin", 0.15))
        self.sampling_objective_max = float(
            al_cfg.get("sampling_objective_max", 0.02)
        )
        self.local_sampling_fraction = float(
            al_cfg.get("local_sampling_fraction", 0.70)
        )
        self.local_elite_objective_max = float(
            al_cfg.get("local_elite_objective_max", 0.03)
        )
        self.local_elite_core_only = bool(
            al_cfg.get("local_elite_core_only", False)
        )
        self.local_center_top_k = int(
            al_cfg.get("local_center_top_k", 0)
        )
        self.local_radii = np.asarray(
            al_cfg.get("local_radii", [0.01, 0.025, 0.05]), dtype=float
        )
        self.exploitation_fraction = float(
            al_cfg.get("exploitation_fraction", 0.75)
        )
        self.uncertainty_penalty = float(
            al_cfg.get("uncertainty_penalty", 0.50)
        )
        self.promising_quantile = float(
            al_cfg.get("promising_quantile", 0.25)
        )
        self.preserve_focused_hybrid = bool(
            al_cfg.get("preserve_focused_hybrid", True)
        )
        self.direct_objective_weight = float(
            al_cfg.get("direct_objective_weight", 0.65)
        )
        self.direct_objective_max = float(
            al_cfg.get("direct_objective_max", 0.25)
        )
        self.direct_objective_core_weight = float(
            al_cfg.get("direct_objective_core_weight", 1.0)
        )
        self.direct_objective_use_robust_target = bool(
            al_cfg.get("direct_objective_use_robust_target", False)
        )
        self.direct_objective_robust_std_weight = float(
            al_cfg.get("direct_objective_robust_std_weight", 1.0)
        )
        self.direct_objective_failure_penalty = float(
            al_cfg.get("direct_objective_failure_penalty", 0.25)
        )
        self.selection_mode = str(
            al_cfg.get("selection_mode", "hybrid")
        ).lower()
        self.diverse_shortlist_quantile = float(
            al_cfg.get("diverse_shortlist_quantile", 0.10)
        )
        self.stability_model_path = al_cfg.get("stability_model_path")
        self.stability_penalty = float(
            al_cfg.get("stability_penalty", 0.0)
        )
        self.minimum_stability_probability = float(
            al_cfg.get("minimum_stability_probability", 0.0)
        )

        self.seed = config["optimization"].get("random_seed", 42)

        # -- Free parameter space ---------------------------------------------
        self.param_space = ForceFieldOptimizer._build_param_space(config)
        self.param_names = [p[0] for p in self.param_space]
        self.param_lo    = np.array([p[1] for p in self.param_space])
        self.param_hi    = np.array([p[2] for p in self.param_space])
        self.n_params    = len(self.param_names)
        self.sample_lo   = self.param_lo.copy()
        self.sample_hi   = self.param_hi.copy()
        self.stability_bundle = self._load_stability_model()

        # -- LAMMPS runner -----------------------------------------------------
        self.runner = LAMMPSRunner(config)

        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(config, output_dir)

        # -- History (accumulated across rounds, written incrementally) -------
        self._history: List[Dict] = []

    # ====================================================================== #
    # Main entry point                                                        #
    # ====================================================================== #

    def _restore_progress(self, initial_data: pd.DataFrame):
        """Recover the last fully recorded AL round and its trained model."""
        all_data = initial_data
        start_round = 1
        model_dir = self.nn_dir
        early_stopped = False
        history_path = os.path.join(
            self.output_dir, "active_learning_history.json"
        )
        if not self.resume or not os.path.exists(history_path):
            return all_data, start_round, model_dir, early_stopped

        with open(history_path, encoding="utf-8") as handle:
            history_document = json.load(handle)
        self._history = list(history_document.get("rounds", []))
        completed_rounds = [
            int(entry["round"])
            for entry in self._history
            if int(entry.get("n_new_evals", 0)) > 0
            and os.path.exists(os.path.join(
                self.output_dir,
                f"data_round_{int(entry['round'])}",
                "all_results.csv",
            ))
        ]
        if completed_rounds:
            last_round = max(completed_rounds)
            data_path = os.path.join(
                self.output_dir,
                f"data_round_{last_round}",
                "all_results.csv",
            )
            all_data = pd.read_csv(data_path)
            round_model = os.path.join(
                self.output_dir, f"nn_round_{last_round}"
            )
            if os.path.exists(os.path.join(round_model, "forward_nn.pt")):
                model_dir = round_model
            start_round = last_round + 1
            print(
                f"  Resumed completed AL round {last_round}: "
                f"{len(all_data)} accumulated rows"
            )
        early_stopped = bool(history_document.get("early_stopped", False))
        if early_stopped or int(history_document.get("final_round", 0)) >= self.n_rounds:
            start_round = self.n_rounds + 1
            print("  AL history is already complete; rebuilding final artifacts only")
        return all_data, start_round, model_dir, early_stopped

    def run(self):
        t0 = time.time()
        mf  = self.config["manifest"]
        al  = self.config["active_learning"]

        print()
        print("=" * 70)
        print("Active Learning Refinement v8")
        print(f"  System       : {mf['system_name']}")
        print(f"  Output dir   : {self.output_dir}")
        print(f"  BO data dir  : {self.bo_dir}")
        print(f"  NN dir       : {self.nn_dir}")
        print(f"  Rounds       : {self.n_rounds}")
        print(f"  Candidates   : {self.n_candidates_per_round} new evals / round")
        print(f"  Pool size    : {self.n_candidate_pool}")
        print(f"  Sampling     : {self.candidate_sampling}")
        print(f"  Threshold    : {self.uncertainty_threshold}  (max ensemble std)")
        print(f"  Obj bias     : {self.objective_bias}  (avoid poor predicted basins)")
        print(f"  Resume       : {self.resume}")
        print("=" * 70)

        # -- Load initial BO data ---------------------------------------------
        print("\nLoading initial BO data")
        all_data_df = self._load_bo_csv()
        n_bo = len(all_data_df)
        print(f"  BO valid rows: {n_bo}")

        all_data_df, start_round, model_dir, early_stopped = (
            self._restore_progress(all_data_df)
        )

        self._configure_sampling_bounds(all_data_df)

        # Snapshot round-0 data (BO only, no AL yet)
        round0_dir = os.path.join(self.output_dir, "data_round_0")
        round0_path = os.path.join(round0_dir, "all_results.csv")
        if start_round == 1 or not os.path.exists(round0_path):
            os.makedirs(round0_dir, exist_ok=True)
            all_data_df.to_csv(round0_path, index=False)

        # -- Load initial ensemble --------------------------------------------
        print("\nLoading initial ensemble")
        pt_path = os.path.join(model_dir, "forward_nn.pt")
        hybrid_manifest = os.path.join(model_dir, "hybrid_manifest.json")
        use_focused_hybrid = os.path.exists(hybrid_manifest)
        if not os.path.exists(pt_path):
            if use_focused_hybrid:
                print(f"  Using focused hybrid model: {self.nn_dir}")
            else:
            # Auto-discover newest nn_output_*/forward_nn.pt
                found = sorted(
                    _glob.glob("nn_output_*/forward_nn.pt"),
                    key=os.path.getmtime, reverse=True
                )
                if found:
                    pt_path = found[0]
                    print(f"  Auto-discovered: {pt_path}")
                else:
                    print(f"  ERROR: {pt_path} not found and no focused hybrid found.")
                    sys.exit(1)

        if use_focused_hybrid:
            local_module_dir = str(Path(__file__).resolve().parents[1] / "analysis")
            if local_module_dir not in sys.path:
                sys.path.insert(0, local_module_dir)
            from focused_hybrid_surrogate import FocusedHybridSurrogate
            hybrid = FocusedHybridSurrogate(model_dir)
            ensembles = hybrid.as_property_ensembles()
            feat_builder = None
            param_names = hybrid.parameter_columns
            feat_names = list(param_names)
            meta = {
                "model_kind": "focused_residual_hybrid",
                "target_names": hybrid.properties,
                "ensemble_size": len(hybrid.multitask_models),
                "input_dim": len(param_names),
                "physics_features": False,
            }
        else:
            ensembles, feat_builder, param_names, feat_names, meta = \
                load_ensemble_from_file(pt_path)

        # Verify param_names match (catches config changes)
        if param_names != self.param_names:
            print(f"  WARNING: loaded model param_names differ from config!")
            print(f"    model  : {param_names}")
            print(f"    config : {self.param_names}")

        target_names = meta.get("target_names", list(self.config["targets"].keys()))
        print(f"  Loaded: {len(ensembles)} property ensembles x "
              f"{meta['ensemble_size']} members")
        print(f"  Properties: {target_names}")
        print(f"  Features: {meta['input_dim']}  "
              f"(physics_features={meta['physics_features']})")

        # -- Round loop -------------------------------------------------------
        physics_enabled = meta["physics_features"]
        derived_charge_enabled = bool(
            meta.get("derived_charge_feature", False)
        )
        completed_objectives = [
            float(entry["best_lammps_obj_so_far"])
            for entry in self._history
            if entry.get("best_lammps_obj_so_far") is not None
            and np.isfinite(float(entry["best_lammps_obj_so_far"]))
        ]
        best_lammps_obj = min(completed_objectives, default=float("inf"))
        last_nn_round_dir: Optional[str] = model_dir

        # Per-property target reference values for uncertainty normalisation
        targets_cfg = self.config["targets"]

        for round_idx in range(start_round, self.n_rounds + 1):
            round_t0 = time.time()
            print(f"\n{'-'*60}")
            print(f"AL Round {round_idx}/{self.n_rounds}")
            print(f"{'-'*60}")

            # -- 1. Sample candidate pool --------------------------------------
            print(f"  [1] Sampling {self.n_candidate_pool} candidates "
                  f"({self.candidate_sampling})")
            X_pool = self._sample_candidate_pool(
                all_data_df,
                self.n_candidate_pool,
                seed_offset=round_idx * 10000
            )
            existing_success = all_data_df
            if "success" in existing_success.columns:
                success_mask = (
                    existing_success["success"].astype(str).str.lower().eq("true")
                )
                existing_success = existing_success.loc[success_mask]
            existing_values = existing_success[self.param_names].to_numpy(dtype=float)
            existing_keys = set(map(tuple, np.round(existing_values, 12)))
            pool_keys = list(map(tuple, np.round(X_pool, 12)))
            previously_evaluated = np.asarray([
                key in existing_keys for key in pool_keys
            ], dtype=bool)
            if previously_evaluated.any():
                print(
                    f"  [1] Excluding {int(previously_evaluated.sum())} "
                    "previously successful candidates"
                )
            pool_param_dicts = self._X_to_param_dicts(X_pool)
            feasible = np.asarray([
                self.runner.feasibility_error(params) is None
                for params in pool_param_dicts
            ], dtype=bool)
            print(
                f"  [1] Feasible candidates: {int(feasible.sum())}/"
                f"{len(feasible)} after charge/derived-parameter constraints"
            )
            stability_probability = self._predict_stability_probability(X_pool)
            if stability_probability is not None:
                print(
                    "  [1] Learned stability probability: "
                    f"min={stability_probability.min():.3f} "
                    f"median={np.median(stability_probability):.3f} "
                    f"max={stability_probability.max():.3f}"
                )

            # -- 2. Rank by aggregated normalised uncertainty ------------------
            # Uncertainty = sum_prop  std_prop(x) / |target_prop|
            # Each property NN contributes independently; normalisation ensures
            # all properties are on comparable scale.
            print(f"  [2] Computing aggregated uncertainty ({len(ensembles)} properties)")
            if physics_enabled:
                X_pool_feat = feat_builder.transform(X_pool)
            elif derived_charge_enabled:
                X_pool_feat = feat_builder.transform_raw_with_derived_charge(X_pool)
            else:
                X_pool_feat = X_pool

            property_std_pool = np.zeros(self.n_candidate_pool)
            for prop, ens in ensembles.items():
                target_val = abs(float(targets_cfg.get(prop, {}).get("value", 1.0)))
                prop_std   = ens.predict_std(X_pool_feat)   # (n_pool,)
                property_std_pool += prop_std / (target_val + 1e-10)
            # Keep the stopping metric tied to the property ensemble only.
            # The acquisition score may additionally include uncertainty from
            # the direct-objective model, but pre/post convergence values must
            # use the same definition.
            std_pool = property_std_pool.copy()

            # mean_pool: weighted RMSE prediction for display only
            # Mirrors lammps_interface._compute_objective: sqrt(sum wx(|e|/t)2 / sumw)
            sq_sum_pool  = np.zeros(self.n_candidate_pool)
            weight_sum_p = 0.0
            for prop, ens in ensembles.items():
                target_val = float(targets_cfg.get(prop, {}).get("value", 1.0))
                weight     = float(targets_cfg.get(prop, {}).get("weight", 1.0))
                pred       = ens.predict_mean(X_pool_feat)
                rel_err    = np.abs(pred - target_val) / (abs(target_val) + 1e-10)
                sq_sum_pool  += weight * rel_err ** 2
                weight_sum_p += weight
            mean_pool = np.sqrt(sq_sum_pool / max(weight_sum_p, 1e-12))

            # A direct objective model complements the property surrogate. It
            # sees the complete LAMMPS objective, including targets deliberately
            # excluded from NN training (currently the noisy cell angles).
            direct_model = self._fit_direct_objective_model(all_data_df, round_idx)
            if direct_model is not None:
                direct_mean, direct_std = self._predict_direct_objective(
                    direct_model, X_pool
                )
                blend = np.clip(self.direct_objective_weight, 0.0, 1.0)
                mean_pool = blend * direct_mean + (1.0 - blend) * mean_pool
                std_pool = np.sqrt(
                    std_pool ** 2 + (blend * direct_std) ** 2
                )
                print(
                    f"  [2] Direct-objective blend={blend:.2f}; "
                    f"train objective<={self.direct_objective_max}"
                )

            max_std  = float(property_std_pool.max())
            mean_std = float(property_std_pool.mean())
            print(f"  [2] max_std={max_std:.6f}  mean_std={mean_std:.6f}")

            # Early stopping check
            if max_std < self.uncertainty_threshold:
                print(f"\n  Early stopping: max_std={max_std:.6f} < "
                      f"threshold={self.uncertainty_threshold}")
                early_stopped = True
                # Log the early-stopped round without new LAMMPS evals
                self._history.append({
                    "round":                  round_idx,
                    "early_stopped":          True,
                    "max_std_before":         max_std,
                    "uncertainty_threshold":  self.uncertainty_threshold,
                    "timestamp":              datetime.now().isoformat(),
                })
                self._save_history()
                break

            # -- 3. Select informative candidates, biased away from poor regions -
            k = min(self.n_candidates_per_round, self.n_candidate_pool)
            std_norm = (std_pool - std_pool.min()) / (np.ptp(std_pool) + 1e-12)
            obj_norm = (mean_pool - mean_pool.min()) / (np.ptp(mean_pool) + 1e-12)
            selectable = feasible & ~previously_evaluated
            if stability_probability is not None:
                selectable &= (
                    stability_probability
                    >= self.minimum_stability_probability
                )
            k = min(k, int(selectable.sum()))
            if k == 0:
                raise RuntimeError("No feasible unevaluated AL candidates remain")

            # Most candidates minimize a conservative objective estimate. This
            # avoids the unrealistically tiny predicted objectives produced by
            # an overconfident surrogate near the edge of its learned domain.
            conservative_obj = mean_pool + self.uncertainty_penalty * std_pool
            if stability_probability is not None:
                conservative_obj = (
                    conservative_obj
                    + self.stability_penalty
                    * (1.0 - stability_probability)
                )
            if self.selection_mode == "diverse_conservative":
                top_idx = self._select_diverse_conservative(
                    X_pool, selectable, conservative_obj, k
                )
            else:
                n_exploit = min(
                    k, max(1, int(round(k * self.exploitation_fraction)))
                )
                exploit_order = np.argsort(conservative_obj)
                exploit_idx = [
                    i for i in exploit_order if selectable[i]
                ][:n_exploit]

                # Keep a smaller uncertainty tranche, but only inside the
                # model's promising objective quantile.
                remaining = selectable.copy()
                remaining[np.asarray(exploit_idx, dtype=int)] = False
                promising_cut = float(np.quantile(
                    mean_pool[selectable], self.promising_quantile
                ))
                explore_score = std_norm - self.objective_bias * obj_norm
                explore_score[
                    ~(remaining & (mean_pool <= promising_cut))
                ] = -np.inf
                n_explore = k - len(exploit_idx)
                explore_idx = np.argsort(
                    explore_score
                )[-n_explore:][::-1].tolist()
                explore_idx = [
                    i for i in explore_idx if np.isfinite(explore_score[i])
                ]

                selected_idx = exploit_idx + explore_idx
                if len(selected_idx) < k:
                    fallback = [
                        i for i in exploit_order
                        if remaining[i] and i not in selected_idx
                    ]
                    selected_idx.extend(fallback[:k - len(selected_idx)])
                top_idx = np.asarray(selected_idx[:k], dtype=int)
            X_selected = X_pool[top_idx]                   # (k, n_params)
            std_sel    = std_pool[top_idx]
            mean_sel   = mean_pool[top_idx]

            print(f"  [3] Selected {k} candidates  "
                  f"std=[{std_sel.min():.4f}, {std_sel.max():.4f}]  "
                  f"pred_obj=[{mean_sel.min():.4f}, {mean_sel.max():.4f}]")

            # -- 4. LAMMPS evaluation ------------------------------------------
            param_dicts = self._X_to_param_dicts(X_selected)
            eval_dir    = os.path.join(
                self.output_dir, f"lammps_round_{round_idx}")
            os.makedirs(eval_dir, exist_ok=True)

            print(f"  [4] LAMMPS: {k} evaluations -> {eval_dir}/")
            results = self.runner.evaluate_batch(
                param_dicts, eval_dir, save_traj=self.save_traj)

            n_valid = sum(1 for r in results if r.success)
            print(f"  [4] {n_valid}/{k} successful")

            # Update best LAMMPS objective seen so far
            for r in results:
                if r.success and np.isfinite(r.objective):
                    if r.objective < best_lammps_obj:
                        best_lammps_obj = r.objective

            # -- 5. Append new evaluations to accumulated dataset --------------
            new_rows = self._results_to_rows(
                results, round_idx, mean_sel, std_sel,
                conservative_obj[top_idx],
                (
                    stability_probability[top_idx]
                    if stability_probability is not None else None
                ),
            )
            new_df   = pd.DataFrame(new_rows)
            all_data_df = pd.concat([all_data_df, new_df],
                                    ignore_index=True)

            # Save accumulated data for this round
            data_round_dir = os.path.join(
                self.output_dir, f"data_round_{round_idx}")
            os.makedirs(data_round_dir, exist_ok=True)
            all_data_df.to_csv(
                os.path.join(data_round_dir, "all_results.csv"), index=False)

            n_total_valid = (
                int(all_data_df["success"].astype(str).str.lower().eq("true").sum())
                if "success" in all_data_df.columns else len(all_data_df)
            )

            # -- 6. Retrain NNSurrogate on expanded dataset --------------------
            print(f"  [5] Retraining NNSurrogate on {n_total_valid} valid points")
            nn_round_dir = os.path.join(
                self.output_dir, f"nn_round_{round_idx}")

            keep_focused = use_focused_hybrid and self.preserve_focused_hybrid
            if keep_focused:
                print("  [5] Preserving focused hybrid between AL rounds; "
                      "generic retraining is skipped")
                last_nn_round_dir = self.nn_dir
            else:
                surrogate = NNSurrogate(
                    config=self.config,
                    bo_dir=data_round_dir,
                    output_dir=nn_round_dir,
                    no_validate=(self.no_validate and round_idx < self.n_rounds),
                    save_traj=self.save_traj,
                    # AL already ranks a constrained candidate pool below.
                    # Running the unconstrained NN optimizer here is redundant,
                    # expensive for hybrid models, and can propose off-manifold
                    # false minima.
                    skip_optimize=True,
                )
                surrogate.run()
                last_nn_round_dir = nn_round_dir

            # -- 7. Load updated per-property ensembles for next round --------
            new_pt = os.path.join(nn_round_dir, "forward_nn.pt")
            if keep_focused:
                print("  [6] Continuing with focused hybrid ensembles")
            elif os.path.exists(new_pt):
                ensembles, feat_builder, param_names, feat_names, meta = \
                    load_ensemble_from_file(new_pt)
                physics_enabled = meta["physics_features"]
                derived_charge_enabled = bool(
                    meta.get("derived_charge_feature", False)
                )
                print(f"  [6] Loaded updated ensembles ({len(ensembles)} props) "
                      f"from {new_pt}")
            else:
                print(f"  WARNING: {new_pt} not found; keeping previous ensembles")

            # -- 8. Extract round metrics --------------------------------------
            # After retraining: re-compute aggregated pool uncertainty
            if physics_enabled:
                X_pool_feat_new = feat_builder.transform(X_pool)
            elif derived_charge_enabled:
                X_pool_feat_new = feat_builder.transform_raw_with_derived_charge(
                    X_pool
                )
            else:
                X_pool_feat_new = X_pool
            std_pool_new = np.zeros(self.n_candidate_pool)
            for prop, ens in ensembles.items():
                tv = abs(float(targets_cfg.get(prop, {}).get("value", 1.0)))
                std_pool_new += ens.predict_std(X_pool_feat_new) / (tv + 1e-10)
            max_std_new  = float(std_pool_new.max())
            mean_std_new = float(std_pool_new.mean())

            # Extract test_r2 from the newly trained surrogate's json
            test_r2 = float("nan")
            opt_json_path = os.path.join(nn_round_dir, "nn_optimize_result.json")
            if keep_focused:
                test_r2 = float("nan")
            elif os.path.exists(opt_json_path):
                with open(opt_json_path) as fh:
                    opt_data = json.load(fh)
                test_r2 = opt_data.get("test_r2", float("nan"))
                if not np.isfinite(test_r2):
                    vals = [
                        float(m.get("r2", np.nan))
                        for m in opt_data.get("test_metrics", {}).values()
                    ]
                    vals = [v for v in vals if np.isfinite(v)]
                    test_r2 = float(np.mean(vals)) if vals else float("nan")

            round_time = time.time() - round_t0

            round_entry = {
                "round":                   round_idx,
                "n_new_evals":             k,
                "n_valid_new":             n_valid,
                "n_total_rows":            len(all_data_df),
                "max_std_before":          max_std,
                "mean_std_before":         mean_std,
                "max_std_after":           max_std_new,
                "mean_std_after":          mean_std_new,
                "best_lammps_obj_so_far":  best_lammps_obj,
                "test_r2":                 test_r2,
                "round_time_s":            round_time,
                "early_stopped":           False,
                "timestamp":               datetime.now().isoformat(),
            }
            self._history.append(round_entry)
            self._save_history()

            print(f"  Done  round={round_idx}  "
                  f"max_std: {max_std:.4f}->{max_std_new:.4f}  "
                  f"R2={test_r2:.4f}  "
                  f"best_obj={best_lammps_obj:.6f}  "
                  f"time={round_time:.0f}s")

            # Convergence check after retraining
            if max_std_new < self.uncertainty_threshold:
                print(f"\n  Post-retrain convergence: "
                      f"max_std={max_std_new:.6f} < "
                      f"threshold={self.uncertainty_threshold}")
                early_stopped = True
                break

        # -- Write nn_round_final (priority-1 target for run.py) -------------
        final_dir = os.path.join(self.output_dir, "nn_round_final")
        if last_nn_round_dir and os.path.isdir(last_nn_round_dir):
            if os.path.isdir(final_dir):
                shutil.rmtree(final_dir)
            shutil.copytree(last_nn_round_dir, final_dir)
            print(f"\n  Copied {last_nn_round_dir} -> {final_dir}")

        final_parameters = self._write_final_parameters(all_data_df)
        best_lammps_obj = float(final_parameters["objective"])

        # -- Final summary ----------------------------------------------------
        elapsed = time.time() - t0
        final_round = self._history[-1]["round"] if self._history else 0
        total_al_evals = sum(e.get("n_new_evals", 0) for e in self._history)

        self._print_summary(final_round, total_al_evals, best_lammps_obj,
                            early_stopped, elapsed)

    # ====================================================================== #
    # Data loading                                                            #
    # ====================================================================== #

    def _read_scored_csv(self, path: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        validate_objective_provenance(
            frame, active_targets(self.config), str(path)
        )
        return frame

    def _load_bo_csv(self) -> pd.DataFrame:
        """
        Load BO data as a DataFrame, preferring stable_results.csv.

        Returns only successful rows (success=True, finite objective).
        Keeps all columns so they can be combined with new AL rows.
        """
        stable_path = os.path.join(self.bo_dir, "stable_results.csv")
        raw_path = os.path.join(self.bo_dir, "all_results.csv")
        training_cfg = self.config.get("nn", {}).get("training_data", {})
        if training_cfg.get("mode") == "core_buffer" and os.path.exists(raw_path):
            raw = self._read_scored_csv(raw_path)
            if "_nn_core_source" not in raw.columns:
                raw["_nn_core_source"] = False
            frames = [raw]
            if os.path.exists(stable_path):
                stable = self._read_scored_csv(stable_path)
                stable["_nn_core_source"] = True
                frames.append(stable)

            roots = [Path.cwd(), Path(self.bo_dir), Path(self.bo_dir).parent]
            seen = set()
            for specification in training_cfg.get("additional_core_files", []) or []:
                candidate = Path(specification)
                patterns = [str(candidate)] if candidate.is_absolute() else [
                    str(root / candidate) for root in roots
                ]
                for pattern in patterns:
                    for match in _glob.glob(pattern):
                        absolute = str(Path(match).resolve())
                        if absolute in seen:
                            continue
                        seen.add(absolute)
                        extra = self._read_scored_csv(absolute)
                        extra["_nn_core_source"] = True
                        existing_core = pd.concat(frames, ignore_index=True, sort=False)
                        if "_nn_core_source" in existing_core.columns:
                            core_marker = (
                                existing_core["_nn_core_source"]
                                .astype(str).str.lower().eq("true")
                            )
                            existing_core = existing_core.loc[core_marker]
                        if len(existing_core) and all(
                                name in existing_core.columns for name in self.param_names):
                            existing_keys = set(map(
                                tuple,
                                np.round(
                                    existing_core[self.param_names].to_numpy(float), 12
                                ),
                            ))
                            extra_keys = list(map(
                                tuple,
                                np.round(extra[self.param_names].to_numpy(float), 12),
                            ))
                            extra = extra.loc[
                                [key not in existing_keys for key in extra_keys]
                            ].copy()
                        frames.append(extra)
            df = pd.concat(frames, ignore_index=True, sort=False)
            print(
                f"  core_buffer AL seed: raw={len(raw)} "
                f"explicit_core={sum(len(frame) for frame in frames[1:])}"
            )
        else:
            csv_path = stable_path if os.path.exists(stable_path) else raw_path
            if not os.path.exists(csv_path):
                sysname = self.config["manifest"]["system_name"]
                found = sorted(
                    _glob.glob(f"bo_{sysname}_*/stable_results.csv"),
                    key=os.path.getmtime, reverse=True
                )
                if not found:
                    found = sorted(
                        _glob.glob(f"bo_{sysname}_*/all_results.csv"),
                        key=os.path.getmtime, reverse=True
                    )
                if not found:
                    found = sorted(
                        _glob.glob("bo_*/stable_results.csv"),
                        key=os.path.getmtime, reverse=True
                    )
                if not found:
                    found = sorted(
                        _glob.glob("bo_*/all_results.csv"),
                        key=os.path.getmtime, reverse=True
                    )
                if found:
                    csv_path = found[0]
                    print(f"  Auto-discovered: {csv_path}")
                else:
                    print(f"  ERROR: {csv_path} not found.")
                    sys.exit(1)
            df = self._read_scored_csv(csv_path)

        if "success" in df.columns:
            success = df["success"].astype(str).str.lower().eq("true")
            df = df.loc[success].copy()
        if "objective" in df.columns:
            df = df[np.isfinite(df["objective"].values)].copy()

        # Ensure param columns are present (fill NaN if missing)
        for name in self.param_names:
            if name not in df.columns:
                df[name] = np.nan

        return df.reset_index(drop=True)

    # ====================================================================== #
    # Candidate sampling                                                      #
    # ====================================================================== #

    def _sample_candidates(self,
                           n_pool:      int,
                           seed_offset: int = 0) -> np.ndarray:
        """
        Generate n_pool candidate parameter vectors in physical units.

        Parameters
        ----------
        n_pool       : number of candidate points to generate
        seed_offset  : added to self.seed so each round uses a different
                       quasi-random sequence; prevents repeated candidates

        Returns
        -------
        X_pool : np.ndarray, shape (n_pool, n_params)
            Candidate raw BO parameter values.
        """
        method = self.candidate_sampling.lower()
        seed   = self.seed + seed_offset

        if method == "sobol":
            sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=seed)
            # Sobol requires n to be a power of 2; round up
            n_sobol = 2 ** math.ceil(math.log2(max(n_pool, 2)))
            unit    = sampler.random(n_sobol)[:n_pool]
        elif method == "lhs":
            sampler = qmc.LatinHypercube(d=self.n_params, seed=seed)
            unit    = sampler.random(n_pool)
        else:   # "random"
            rng  = np.random.default_rng(seed)
            unit = rng.random((n_pool, self.n_params))

        return qmc.scale(unit, self.sample_lo, self.sample_hi)

    def _sample_candidate_pool(
            self, data: pd.DataFrame, n_pool: int,
            seed_offset: int = 0) -> np.ndarray:
        """Mix envelope-wide candidates with local perturbations of elites."""
        n_local = int(round(n_pool * np.clip(self.local_sampling_fraction, 0.0, 1.0)))
        n_global = n_pool - n_local
        global_pool = self._sample_candidates(n_global, seed_offset)
        if n_local == 0:
            return global_pool

        elite = data.copy()
        if "success" in elite.columns:
            elite = elite.loc[
                elite["success"].astype(str).str.lower().eq("true")
            ]
        if self.local_elite_core_only and "_nn_core_source" in elite.columns:
            core = elite["_nn_core_source"].astype(str).str.lower().eq("true")
            elite = elite.loc[core]
        elite = elite.loc[
            pd.to_numeric(elite["objective"], errors="coerce")
            <= self.local_elite_objective_max
        ]
        finite = np.isfinite(elite[self.param_names].to_numpy(dtype=float)).all(axis=1)
        elite = elite.loc[finite].sort_values("objective")
        if self.local_center_top_k > 0:
            elite = elite.head(self.local_center_top_k)
        if elite.empty:
            print("  [1] No local elites found; using envelope-wide candidates")
            return self._sample_candidates(n_pool, seed_offset)

        centers = elite[self.param_names].to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed + seed_offset + 7919)
        center_idx = rng.integers(0, len(centers), size=n_local)
        radii = rng.choice(self.local_radii, size=n_local)
        span = np.maximum(self.sample_hi - self.sample_lo, 1e-12)
        noise = rng.normal(size=(n_local, self.n_params)) * radii[:, None] * span
        local_pool = np.clip(
            centers[center_idx] + noise, self.sample_lo, self.sample_hi
        )
        print(
            f"  [1] Candidate mix: {n_global} envelope + {n_local} local "
            f"around {len(centers)} elites (objective<="
            f"{self.local_elite_objective_max})"
        )
        return np.vstack([global_pool, local_pool])

    def _fit_direct_objective_model(
            self, data: pd.DataFrame,
            round_idx: int) -> Optional[ExtraTreesRegressor]:
        """Fit a local log-objective ensemble for acquisition ranking only."""
        frame = data.copy()
        if "success" in frame.columns:
            frame = frame.loc[
                frame["success"].astype(str).str.lower().eq("true")
            ]
        objective = pd.to_numeric(frame.get("objective"), errors="coerce")
        core_marker = pd.Series(False, index=frame.index)
        if "_nn_core_source" in frame.columns:
            core_marker = (
                frame["_nn_core_source"].astype(str).str.lower().eq("true")
            )

        robust_rows = pd.Series(False, index=frame.index)
        if (
            self.direct_objective_use_robust_target
            and "objective_mean" in frame.columns
            and "objective_std" in frame.columns
        ):
            robust_mean = pd.to_numeric(
                frame["objective_mean"], errors="coerce"
            )
            robust_std = pd.to_numeric(
                frame["objective_std"], errors="coerce"
            )
            if "failure_rate" in frame.columns:
                failure_rate = pd.to_numeric(
                    frame["failure_rate"], errors="coerce"
                ).fillna(0.0)
            else:
                failure_rate = pd.Series(0.0, index=frame.index)
            robust_rows = (
                core_marker
                & np.isfinite(robust_mean)
                & np.isfinite(robust_std)
            )
            objective.loc[robust_rows] = (
                robust_mean.loc[robust_rows]
                + self.direct_objective_robust_std_weight
                * robust_std.loc[robust_rows]
                + self.direct_objective_failure_penalty
                * failure_rate.loc[robust_rows]
            )

        finite = np.isfinite(objective) & (objective > 0.0)
        finite &= objective <= self.direct_objective_max
        x = frame.loc[finite, self.param_names].to_numpy(dtype=float)
        y = objective.loc[finite].to_numpy(dtype=float)
        sample_weight = np.ones(len(x), dtype=float)
        if self.direct_objective_core_weight != 1.0:
            core_selected = core_marker.loc[finite].to_numpy(dtype=bool)
            sample_weight[core_selected] = self.direct_objective_core_weight
        valid_x = np.isfinite(x).all(axis=1)
        x, y = x[valid_x], y[valid_x]
        sample_weight = sample_weight[valid_x]
        if len(x) < max(50, 2 * self.n_params):
            print(f"  [2] Direct-objective model skipped: only {len(x)} rows")
            return None

        x = (x - self.param_lo) / np.maximum(self.param_hi - self.param_lo, 1e-12)
        model = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=2,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.85,
            n_jobs=-1,
            random_state=self.seed + round_idx,
        )
        model.fit(x, np.log10(y), sample_weight=sample_weight)
        model._ffopt_training_rows = len(x)
        n_core = int(np.count_nonzero(sample_weight != 1.0))
        n_robust = int((robust_rows & finite).sum())
        print(
            f"  [2] Direct-objective ExtraTrees: {len(x)} local rows; "
            f"robust={n_robust}, core_weighted={n_core} "
            f"(weight={self.direct_objective_core_weight:g})"
        )
        return model

    def _predict_direct_objective(
            self, model: ExtraTreesRegressor,
            x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return geometric mean and ensemble spread in objective units."""
        scaled = (x - self.param_lo) / np.maximum(
            self.param_hi - self.param_lo, 1e-12
        )
        tree_log = np.vstack([
            estimator.predict(scaled) for estimator in model.estimators_
        ])
        values = np.power(10.0, tree_log)
        return values.mean(axis=0), values.std(axis=0)

    def _select_diverse_conservative(
            self, x: np.ndarray, selectable: np.ndarray,
            conservative_obj: np.ndarray, k: int) -> np.ndarray:
        """Maximin design within the conservative objective shortlist."""
        available = np.flatnonzero(selectable)
        cutoff = float(np.quantile(
            conservative_obj[available], self.diverse_shortlist_quantile
        ))
        shortlist = available[conservative_obj[available] <= cutoff]
        if len(shortlist) < k:
            shortlist = available

        scaled = (x[shortlist] - self.param_lo) / np.maximum(
            self.param_hi - self.param_lo, 1e-12
        )
        first_local = int(np.argmin(conservative_obj[shortlist]))
        chosen_local = [first_local]
        min_distance = np.linalg.norm(
            scaled - scaled[first_local], axis=1
        )
        min_distance[first_local] = -np.inf
        while len(chosen_local) < min(k, len(shortlist)):
            next_local = int(np.argmax(min_distance))
            chosen_local.append(next_local)
            distance = np.linalg.norm(
                scaled - scaled[next_local], axis=1
            )
            min_distance = np.minimum(min_distance, distance)
            min_distance[np.asarray(chosen_local, dtype=int)] = -np.inf
        print(
            f"  [3] Diverse conservative design: shortlist={len(shortlist)} "
            f"quantile={self.diverse_shortlist_quantile:.3f}"
        )
        return shortlist[np.asarray(chosen_local, dtype=int)]

    def _configure_sampling_bounds(self, data: pd.DataFrame) -> None:
        """Constrain AL proposals to the validated stable-core envelope."""
        if self.sampling_domain != "core_envelope":
            print("  AL sampling domain: global parameter bounds")
            return

        selected = data.copy()
        if "_nn_core_source" in selected.columns:
            marker = selected["_nn_core_source"].astype(str).str.lower().eq("true")
            selected = selected.loc[marker].copy()
        if "objective" in selected.columns:
            selected = selected.loc[
                selected["objective"] <= self.sampling_objective_max
            ].copy()
        finite = np.isfinite(
            selected[self.param_names].to_numpy(dtype=float)
        ).all(axis=1)
        selected = selected.loc[finite]
        if len(selected) < 10:
            print("  WARNING: fewer than 10 core rows; using global AL bounds")
            return

        values = selected[self.param_names].to_numpy(dtype=float)
        lower = np.quantile(values, 0.01, axis=0)
        upper = np.quantile(values, 0.99, axis=0)
        span = np.maximum(upper - lower, 1e-12)
        self.sample_lo = np.maximum(self.param_lo, lower - self.envelope_margin * span)
        self.sample_hi = np.minimum(self.param_hi, upper + self.envelope_margin * span)
        print(
            f"  AL sampling domain: core envelope from {len(selected)} rows "
            f"(objective<={self.sampling_objective_max}) "
            f"with margin={self.envelope_margin:.2f}"
        )

    # ====================================================================== #
    # Helpers                                                                 #
    # ====================================================================== #

    def _X_to_param_dicts(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Convert (N, D) parameter matrix to list of {name: value} dicts."""
        return [
            {n: float(np.clip(X[i, j], self.param_lo[j], self.param_hi[j]))
             for j, n in enumerate(self.param_names)}
            for i in range(X.shape[0])
        ]

    def _results_to_rows(
            self,
            results: list,
            round_idx: int,
            predicted_objective: Optional[np.ndarray] = None,
            predicted_uncertainty: Optional[np.ndarray] = None,
            conservative_score: Optional[np.ndarray] = None,
            predicted_stability: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Convert LAMMPSRunner EvalResult objects to a list of row dicts
        matching the all_results.csv schema written by optimizer.py.
        """
        rows = []
        for i, r in enumerate(results):
            row: Dict = {
                "round":     round_idx,
                "label":     f"al_round_{round_idx}",
                "success":   r.success,
                "objective": r.objective,
                "error_msg": r.error_msg if hasattr(r, "error_msg") else "",
                "al_predicted_objective": (
                    float(predicted_objective[i])
                    if predicted_objective is not None else float("nan")
                ),
                "al_predicted_uncertainty": (
                    float(predicted_uncertainty[i])
                    if predicted_uncertainty is not None else float("nan")
                ),
                "al_conservative_score": (
                    float(conservative_score[i])
                    if conservative_score is not None else float("nan")
                ),
                "al_predicted_stability": (
                    float(predicted_stability[i])
                    if predicted_stability is not None else float("nan")
                ),
            }
            row.update(objective_provenance(active_targets(self.config)))
            # Free BO param values
            for name in self.param_names:
                row[name] = r.params.get(name, float("nan"))
            # Per-target computed values + error percentages
            for prop in self.config["targets"]:
                row[f"calc_{prop}"]  = r.properties.get(prop,  float("nan"))
                row[f"error_{prop}"] = r.per_property_error.get(prop, float("nan"))
            # Auxiliary surface energy decomposition columns
            for aux in ("E_complete", "E_split", "A_xy"):
                row[f"calc_{aux}"] = r.properties.get(aux, float("nan"))
            rows.append(row)
        return rows

    def _load_stability_model(self):
        """Load an optional probability model for post-feasibility stability."""
        if not self.stability_model_path:
            return None
        candidates = [Path(self.stability_model_path)]
        config_path = self.config.get("_config_path")
        if config_path:
            candidates.append(Path(config_path).resolve().parent / self.stability_model_path)
        path = next((item.resolve() for item in candidates if item.exists()), None)
        if path is None:
            print(
                "  WARNING: stability_model_path does not exist; learned "
                f"stability gate disabled: {self.stability_model_path}"
            )
            return None
        import joblib
        bundle = joblib.load(path)
        if bundle.get("parameter_names") != self.param_names:
            raise ValueError(
                "Stability model parameter columns do not match AL parameter space"
            )
        print(f"  Loaded learned stability gate: {path}")
        return bundle

    def _predict_stability_probability(
            self, x: np.ndarray) -> Optional[np.ndarray]:
        if self.stability_bundle is None:
            return None
        lo = np.asarray(self.stability_bundle["parameter_lo"], dtype=float)
        hi = np.asarray(self.stability_bundle["parameter_hi"], dtype=float)
        scaled = (x - lo) / np.maximum(hi - lo, 1e-12)
        model = self.stability_bundle["model"]
        probabilities = model.predict_proba(scaled)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Stability classifier has no stable class label 1")
        return probabilities[:, classes.index(1)]

    # ====================================================================== #
    # History I/O                                                             #
    # ====================================================================== #

    def _write_final_parameters(self, frame: pd.DataFrame) -> dict:
        """Export the best successful point that LAMMPS actually evaluated."""
        objective = pd.to_numeric(frame.get("objective"), errors="coerce")
        valid = np.isfinite(objective)
        if "success" in frame.columns:
            valid &= frame["success"].astype(str).str.lower().eq("true")
        for name in self.param_names:
            if name not in frame.columns:
                raise ValueError(f"Accumulated AL data is missing parameter {name!r}")
            valid &= np.isfinite(pd.to_numeric(frame[name], errors="coerce"))
        candidates = frame.loc[valid].copy()
        if candidates.empty:
            raise RuntimeError("No successful LAMMPS-evaluated parameters are available")
        candidates["objective"] = pd.to_numeric(
            candidates["objective"], errors="raise"
        )
        best = candidates.sort_values("objective").iloc[0]
        raw = {name: float(best[name]) for name in self.param_names}
        label = best.get("label")
        if pd.isna(label) or not str(label).strip():
            label = "sampling" if pd.notna(best.get("candidate_id")) else "bo"
        candidate_id = best.get("candidate_id")
        properties = {}
        for name in self.config.get("targets", {}):
            value = pd.to_numeric(best.get(f"calc_{name}"), errors="coerce")
            if np.isfinite(value):
                properties[name] = float(value)
        document = {
            "selection_rule": (
                "minimum objective among successful accumulated "
                "LAMMPS evaluations"
            ),
            "objective": float(best["objective"]),
            "source_round": (
                int(best["round"])
                if "round" in best and pd.notna(best["round"])
                else None
            ),
            "source_label": str(label),
            "source_candidate_id": (
                int(candidate_id) if pd.notna(candidate_id) else None
            ),
            "raw_free_parameters": raw,
            "properties": properties,
            "timestamp": datetime.now().isoformat(),
        }
        path = Path(self.output_dir) / "final_parameters.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        print(
            f"  Best verified parameters: objective={document['objective']:.6f} "
            f"-> {path}"
        )
        return document

    def _save_history(self):
        """
        Write active_learning_history.json incrementally (after every round).

        Format consumed by viz.plot_active_learning_curve:
        {
          "rounds":       [ { round, n_new_evals, max_std_before, ... }, ... ],
          "early_stopped": bool,
          "final_round":  int,
          "total_al_evals": int
        }
        """
        final_round    = self._history[-1]["round"] if self._history else 0
        total_al_evals = sum(e.get("n_new_evals", 0) for e in self._history)
        early_stopped  = any(e.get("early_stopped", False)
                             for e in self._history)

        doc = {
            "rounds":        self._history,
            "early_stopped": early_stopped,
            "final_round":   final_round,
            "total_al_evals": total_al_evals,
            "timestamp":     datetime.now().isoformat(),
        }
        path = os.path.join(self.output_dir, "active_learning_history.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)

    # ====================================================================== #
    # Final summary                                                           #
    # ====================================================================== #

    def _print_summary(self,
                       final_round:   int,
                       total_al_evals: int,
                       best_obj:      float,
                       early_stopped: bool,
                       elapsed:       float):
        print()
        print("=" * 70)
        print("Active Learning Complete")
        print("=" * 70)
        print(f"  Rounds completed : {final_round} / {self.n_rounds}")
        print(f"  Early stopped    : {early_stopped}")
        print(f"  Total AL evals   : {total_al_evals}")
        print(f"  Best LAMMPS obj  : {best_obj:.6f}")
        print(f"  Elapsed          : {elapsed/60:.1f} min")
        print()
        print("  Output files:")
        print(f"    {self.output_dir}/nn_round_final/nn_optimize_result.json")
        print(f"    {self.output_dir}/active_learning_history.json")
        print()
        print("  Next steps:")
        print("    ffopt validate               # validate best params")
        print("    ffopt plot                   # generate all figures")
        print("=" * 70)


# ============================================================================
# math import (needed by _sample_candidates for Sobol power-of-2 rounding)
# ============================================================================
import math


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Active Learning Refinement v8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the generated internal JSON configuration.")
    parser.add_argument(
        "--bo-dir", default=None,
        help="BO results directory (stable_results.csv preferred, "
             "all_results.csv fallback). "
             "Default: auto-discover newest bo_*/.")
    parser.add_argument(
        "--nn-dir", default=None,
        help="NNSurrogate output directory (must contain forward_nn.pt). "
             "Default: auto-discover newest nn_output_*/.")
    parser.add_argument(
        "--output-dir", default="active_learning_output",
        help="Output directory (default: active_learning_output/)")
    parser.add_argument(
        "--model",
        choices=["mlp_ensemble", "gp", "random_forest", "xgboost"],
        default=None,
        help="Override nn.model for AL retraining without editing config.")
    parser.add_argument(
        "--train-top-fraction", type=float, default=None,
        help="Override nn.train_top_fraction for AL retraining.")
    parser.add_argument(
        "--additional-core-file", action="append", default=None,
        help="Additional stability-mean CSV retained across AL retraining.")
    parser.add_argument("--core-objective-max", type=float, default=None)
    parser.add_argument("--buffer-objective-max", type=float, default=None)
    parser.add_argument("--core-weight", type=int, default=None)
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip LAMMPS validation in intermediate NN retraining rounds. "
             "Final round always validates.")
    parser.add_argument(
        "--save-traj", action="store_true",
        help="Save LAMMPS trajectory files during evaluation.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue after the last fully recorded AL round.")
    args = parser.parse_args()

    # -- Load config -----------------------------------------------------------
    config = load_expanded_config(args.config)

    if args.model:
        config.setdefault("nn", {})["model"] = args.model
    if args.train_top_fraction is not None:
        config.setdefault("nn", {})["train_top_fraction"] = args.train_top_fraction
    if args.additional_core_file:
        config.setdefault("nn", {}).setdefault("training_data", {})[
            "additional_core_files"
        ] = args.additional_core_file
    training_cfg = config.setdefault("nn", {}).setdefault("training_data", {})
    if args.core_objective_max is not None:
        training_cfg["core_objective_max"] = args.core_objective_max
    if args.buffer_objective_max is not None:
        training_cfg["buffer_objective_max"] = args.buffer_objective_max
    if args.core_weight is not None:
        training_cfg["core_weight"] = args.core_weight

    if not config.get("active_learning", {}).get("enabled", True):
        print("active_learning.enabled: false - skipping AL.")
        sys.exit(0)

    # -- Resolve bo_dir --------------------------------------------------------
    bo_dir = args.bo_dir
    if bo_dir is None:
        sysname = config["manifest"]["system_name"]
        csvs = sorted(_glob.glob(f"bo_{sysname}_*/stable_results.csv"),
                      key=os.path.getmtime, reverse=True)
        if not csvs:
            csvs = sorted(_glob.glob(f"bo_{sysname}_*/all_results.csv"),
                          key=os.path.getmtime, reverse=True)
        if not csvs:
            csvs = sorted(_glob.glob("bo_*/stable_results.csv"),
                          key=os.path.getmtime, reverse=True)
        if not csvs:
            csvs = sorted(_glob.glob("bo_*/all_results.csv"),
                          key=os.path.getmtime, reverse=True)
        if csvs:
            bo_dir = os.path.dirname(csvs[0])
            print(f"Auto-selected BO dir: {bo_dir}")
        else:
            print("ERROR: --bo-dir not given and no bo_*/all_results.csv found.")
            sys.exit(1)

    # -- Resolve nn_dir --------------------------------------------------------
    nn_dir = args.nn_dir
    if nn_dir is None:
        sysname = config["manifest"]["system_name"]
        pts = sorted(_glob.glob(f"nn_output_{sysname}_*/forward_nn.pt"),
                     key=os.path.getmtime, reverse=True)
        if not pts:
            pts = sorted(_glob.glob("nn_output_*/forward_nn.pt"),
                         key=os.path.getmtime, reverse=True)
        if pts:
            nn_dir = os.path.dirname(pts[0])
            print(f"Auto-selected NN dir: {nn_dir}")
        else:
            # Try legacy path without timestamp
            if os.path.exists("nn_output/forward_nn.pt"):
                nn_dir = "nn_output"
                print(f"Using legacy NN dir: {nn_dir}")
            else:
                print("ERROR: --nn-dir not given and no forward_nn.pt found.")
                sys.exit(1)

    # -- Run active learning ---------------------------------------------------
    learner = ActiveLearner(
        config=config,
        bo_dir=bo_dir,
        nn_dir=nn_dir,
        output_dir=args.output_dir,
        no_validate=args.no_validate,
        save_traj=args.save_traj,
        resume=args.resume,
    )
    learner.run()


if __name__ == "__main__":
    main()
