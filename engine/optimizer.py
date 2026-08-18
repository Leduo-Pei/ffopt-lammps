"""
optimizer.py  |  Bayesian Optimizer for Force Field Parameters

BO method selection:
  - If optimization.method is "gp", "turbo", or "saasbo", that user choice is used.
  - If optimization.method is "auto", the code selects by dimensionality:
      accuracy_priority=true:
        n_params < 7        -> GP
        7 <= n_params < 10  -> TuRBO
        n_params >= 10      -> SAASBO
      accuracy_priority=false:
        n_params < 7        -> GP
        7 <= n_params < 16  -> TuRBO
        n_params >= 16      -> SAASBO

Pareto BO:
  obj_structural = weighted RMSE of non-surface targets
  obj_surface    = surface-energy percent error
  active         = qNEHVI when GP + compute_surface
  posthoc        = Pareto front extracted from all valid evaluations after BO
"""

import math
import os
import csv
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config_loader import save_config_snapshot

from utils.objective_rescoring import active_targets, objective_provenance

        # SAASBO
try:
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.fit import fit_fully_bayesian_model_nuts
    _SAASBO_AVAILABLE = True
except ImportError:
    _SAASBO_AVAILABLE = False

# Optional: Pareto (multi-objective) BO requires botorch >= 0.9
try:
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement)
    from botorch.utils.multi_objective.pareto import is_non_dominated
    _PARETO_AVAILABLE = True
except ImportError:
    _PARETO_AVAILABLE = False

from .lammps_interface import LAMMPSRunner, LARGE_PENALTY
from .parameter_space import build_parameter_space

# Suppress repetitive BoTorch / GPyTorch warnings
warnings.filterwarnings("ignore", message=".*qNoisyExpectedImprovement.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch.optim")
warnings.filterwarnings("ignore", message=".*added jitter.*")
warnings.filterwarnings("ignore", message=".*A not p.d..*")

# GP kernel matrix conditioning limit.
# Beyond ~512 points the Cholesky factorisation degrades; keep only the best N.
MAX_GP_POINTS = 512


# ============================================================================
# TuRBO
# ============================================================================

@dataclass
class TurboState:
    """
    Trust-region state for one TuRBO restart.

    The trust region is an axis-aligned hyper-rectangle of side `length`
    (in normalised [0,1] space) centred at the current best point.
    On success: length doubles (expand).  On failure: length halves (contract).
    When length < length_min, the state is flagged for restart.
    """
    dim:              int
    batch_size:       int
    length:           float = 0.8
    length_min:       float = 0.005
    length_max:       float = 1.6
    failure_counter:  int   = 0
    success_counter:  int   = 0
    best_value:       float = float("inf")   # lowest objective seen so far
    restart_triggered: bool = False

    # TuRBO
    failure_tolerance: int = field(init=False)
    success_tolerance: int = 10

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim))
        )


def _update_turbo_state(state: TurboState, new_objectives: List[float]) -> TurboState:
    """
    Update TuRBO state after a batch of evaluations.

    new_objectives : list of objective values from the latest batch
                     (lower is better; LARGE_PENALTY for failures)
    """
    best_new = min(new_objectives)
    # Success: improved beyond a tiny tolerance relative to current best
    if best_new < state.best_value - 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter  = 0
    else:
        state.failure_counter += 1
        state.success_counter  = 0

    # Expand trust region after sustained success
    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0

    # Contract trust region after sustained failure
    if state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = min(state.best_value, best_new)

    if state.length < state.length_min:
        state.restart_triggered = True

    return state


# ============================================================================
# ForceFieldOptimizer
# ============================================================================

class ForceFieldOptimizer:
    """
    Bayesian Optimiser for multi-type LJ (+ Coulomb) force field parameters.

    Usage
    -----
    optimizer = ForceFieldOptimizer(config)
    summary   = optimizer.run()

    Parameters
    ----------
    config : dict
        Generated expanded engine configuration.
    """

    def __init__(self, config: dict):
        self.config = config

        # ------------------------------------------------------------------ #
        # Build free-parameter space from config                              #
        # Mirrors the exclusion logic in lammps_interface._resolve_params()   #
        # ------------------------------------------------------------------ #
        self.param_space = self._build_param_space(config)
        # param_space : list of (name, lo, hi)
        self.param_names = [p[0] for p in self.param_space]
        self.n_params    = len(self.param_names)

        bounds_lo = [p[1] for p in self.param_space]
        bounds_hi = [p[2] for p in self.param_space]
        self.bounds = torch.tensor([bounds_lo, bounds_hi], dtype=torch.double)

        # ------------------------------------------------------------------ #
        # Optimization settings                                               #
        # ------------------------------------------------------------------ #
        opt_cfg = config["optimization"]

        self.n_initial      = opt_cfg["n_initial"]
        self.n_bo_iters     = opt_cfg["n_bo_iterations"]
        # ``batch_size`` is part of the scientific optimization protocol.
        # ``max_workers`` only controls how many of those candidates run at
        # once.  Keeping them separate makes one- and two-node runs evaluate
        # the same candidates and differ only in elapsed time.
        self.batch_size     = max(1, int(opt_cfg.get("batch_size", 48)))
        self.objective_type = opt_cfg.get("objective", "weighted_rmse")

        # TuRBO
        self.bo_method = self._select_bo_method(self.n_params, opt_cfg)

        # TuRBO
        turbo_cfg = opt_cfg.get("turbo", {})
        self.turbo_n_tr       = turbo_cfg.get("n_trust_regions", 1)
        self.turbo_length_init = turbo_cfg.get("length_init", 0.8)
        self.turbo_length_min  = turbo_cfg.get("length_min",  0.005)
        self.turbo_length_max  = turbo_cfg.get("length_max",  1.6)

        # SAASBO
        saasbo_cfg = opt_cfg.get("saasbo", {})
        self.saasbo_num_samples  = saasbo_cfg.get("num_samples",  256)
        self.saasbo_warmup_steps = saasbo_cfg.get("warmup_steps", 512)

        # Early stopping (optional; defaults allow the full run)
        es_cfg = opt_cfg.get("early_stop", {})
        self.early_stop      = es_cfg.get("enabled", True)
        self.patience        = es_cfg.get("patience", 20)
        self.min_improvement = es_cfg.get("min_improvement", 0.001)

        # Exploration rounds (periodic random + uncertainty sampling)
        exp_cfg = opt_cfg.get("exploration", {})
        self.explore_every     = exp_cfg.get("every_n_rounds", 5)
        self.explore_n_random  = exp_cfg.get("n_random", self.batch_size)
        self.explore_n_uncertain = exp_cfg.get("n_uncertain", self.batch_size // 2)

        # Feasibility classifier
        feas_cfg = opt_cfg.get("feasibility", {})
        self.penalty_cap_multiplier     = feas_cfg.get("penalty_cap_multiplier", 3.0)
        self.use_feasibility_classifier = feas_cfg.get("classifier", True)

        self.seed = opt_cfg.get("random_seed", 42)

        # Live BO reporting. The default deliberately prints every round so a
        # cluster user can see which physical property limits the objective
        # without waiting for the final optimization report.
        progress_cfg = opt_cfg.get("progress_report", {})
        self.progress_enabled = bool(progress_cfg.get("enabled", True))
        self.progress_table_every = max(
            1, int(progress_cfg.get("property_table_every", 1)))
        self.progress_save_csv = bool(progress_cfg.get("save_csv", True))
        self._active_targets = active_targets(config)

        # Stability audit checks top candidates across independent NPT seeds
        # before they are used by the surrogate.
        stab_cfg = opt_cfg.get("stability_audit", {})
        self.stability_enabled = stab_cfg.get("enabled", False)
        self.stability_top_k = int(stab_cfg.get("top_k", 0))
        self.stability_seeds = [int(s) for s in stab_cfg.get("seeds", [])]
        self.stability_max_workers = int(
            stab_cfg.get("max_workers", config["parallel"]["max_workers"]))
        self.stability_max_obj_std = float(
            stab_cfg.get("max_objective_std", float("inf")))
        self.stability_max_prop_rel_std = float(
            stab_cfg.get("max_property_rel_std", float("inf")))
        self.stability_noise_penalty = float(
            stab_cfg.get("noise_penalty", 1.0))
        self.stability_failure_penalty = float(
            stab_cfg.get("failure_penalty", 10.0))

        # ------------------------------------------------------------------ #
        # Internal state                                                      #
        # ------------------------------------------------------------------ #
        # GP training data: valid + capped-penalty points
        self.train_X = torch.empty(0, self.n_params, dtype=torch.double)
        self.train_Y = torch.empty(0, 1,             dtype=torch.double)

        # All points (for feasibility classifier)
        self.all_X       = torch.empty(0, self.n_params, dtype=torch.double)
        self.all_feasible: List[bool] = []

        self.all_results:   List[dict]  = []
        self.best_history:  List[float] = []
        self.round_times:   List[float] = []
        self.current_round: int         = 0
        self._best_valid_obj: float     = float("inf")
        self.bo_finished: bool          = False

        self._feasibility_model = None

        # TuRBO
        self._turbo_state: Optional[TurboState] = None

        # ------------------------------------------------------------------ #
        # LAMMPS runner                                                        #
        # ------------------------------------------------------------------ #
        self.runner = LAMMPSRunner(config)

        # ------------------------------------------------------------------ #
        # Work directory and checkpoint                                        #
        # ------------------------------------------------------------------ #
        sys_name  = config["manifest"]["system_name"]
        run_root = config.get("workflow", {}).get("run_root", ".")
        self.run_root = os.path.abspath(os.path.expanduser(str(run_root)))
        os.makedirs(self.run_root, exist_ok=True)
        configured_output = config.get("workflow", {}).get("bo_output_dir")
        if configured_output:
            self.work_dir = os.path.abspath(os.path.expanduser(str(configured_output)))
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = os.path.join(
                self.run_root, f"bo_{sys_name}_{timestamp}"
            )
        os.makedirs(self.work_dir, exist_ok=True)
        save_config_snapshot(config, self.work_dir)

        ckpt_cfg = config.get("checkpoint", {})
        self.ckpt_enabled = ckpt_cfg.get("enabled", True)
        self.ckpt_every   = ckpt_cfg.get("interval", 10)
        self.ckpt_dir     = os.path.join(
            self.work_dir,
            ckpt_cfg.get("directory", "checkpoints")
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Pareto BO configuration                                             #
        # ------------------------------------------------------------------ #
        pareto_cfg        = opt_cfg.get("pareto", {})
        pareto_mode_cfg   = pareto_cfg.get("mode", "auto")
        compute_surface   = config["lammps"]["compute_surface"]
        ref_pt_cfg        = pareto_cfg.get("reference_point", {})

        # Active Pareto: qNEHVI, only meaningful when surf is computed
        self.pareto_active = (
            _PARETO_AVAILABLE
            and compute_surface
            and self.bo_method == "gp"
            and pareto_mode_cfg in ("auto", "active")
        )
        # Post-hoc Pareto: always computed when surface is available
        self.pareto_posthoc = compute_surface

        # Reference point for qNEHVI (negated: BoTorch maximizes)
        struct_ref = float(ref_pt_cfg.get("structural", 30.0)) / 100.0
        surf_ref   = float(ref_pt_cfg.get("surface",    50.0))
        self.pareto_ref_point = torch.tensor(
            [-struct_ref, -surf_ref], dtype=torch.double)

        # 2-column training tensor: [-obj_structural, -obj_surface] (BoTorch maximizes)
        self.train_Y2 = torch.empty(0, 2, dtype=torch.double)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _report_best_progress(
        self,
        stage: str,
        round_time: Optional[float] = None,
        improved: Optional[bool] = None,
    ) -> None:
        """Print and persist the current global-best property comparison."""
        if not self.progress_enabled:
            return

        valid = [
            result for result in self.all_results
            if result.get("success")
            and math.isfinite(float(result.get("objective", float("nan"))))
        ]
        if not valid:
            return

        best = min(valid, key=lambda result: float(result["objective"]))
        status = ""
        if improved is True:
            status = " [IMPROVED]"
        elif improved is False:
            status = " [UNCHANGED]"

        print(
            f"  Current global best target comparison{status}: "
            f"objective={float(best['objective']):.6f}, "
            f"found={best.get('round', best.get('label', '?'))}"
        )
        print(
            f"    {'Property':<14} {'Calculated':>12} "
            f"{'Target':>10} {'Error%':>8}"
        )
        print(f"    {'-' * 46}")

        latest_rows = []
        history_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "round": self.current_round,
            "best_found_round": best.get("round", ""),
            "best_label": best.get("label", ""),
            "objective": float(best["objective"]),
            "improved": improved if improved is not None else "",
            "total_evaluations": len(self.all_results),
            "valid_evaluations": len(valid),
            "round_time_s": round_time if round_time is not None else "",
        }

        for prop, info in self._active_targets.items():
            calc = float(best.get(f"calc_{prop}", float("nan")))
            target = float(info["value"])
            err = float(best.get(f"error_{prop}", float("nan")))
            flag = "OK" if math.isfinite(err) and err < 5.0 else "!!"
            print(
                f"    {prop:<14} {calc:>12.4f} {target:>10.4f} "
                f"{err:>7.2f}%  {flag}"
            )
            latest_rows.append({
                "stage": stage,
                "round": self.current_round,
                "objective": float(best["objective"]),
                "best_found_round": best.get("round", ""),
                "best_label": best.get("label", ""),
                "property": prop,
                "calculated": calc,
                "target": target,
                "error_percent": err,
                "weight": float(info.get("weight", 1.0)),
                "status": flag,
            })
            history_row[f"calc_{prop}"] = calc
            history_row[f"target_{prop}"] = target
            history_row[f"error_{prop}"] = err

        if not self.progress_save_csv:
            return

        latest_path = os.path.join(self.work_dir, "progress_latest.csv")
        with open(latest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(latest_rows[0]))
            writer.writeheader()
            writer.writerows(latest_rows)

        history_path = os.path.join(self.work_dir, "progress_history.csv")
        write_header = not os.path.exists(history_path)
        with open(history_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history_row))
            if write_header:
                writer.writeheader()
            writer.writerow(history_row)

    # ====================================================================== #
    # Static helpers: parameter space + method selection                     #
    # ====================================================================== #

    @staticmethod
    def _build_param_space(config: dict) -> List[Tuple[str, float, float]]:
        """Compatibility wrapper around the lightweight shared builder."""
        return build_parameter_space(config)

    @staticmethod
    def _select_bo_method(n_params: int, opt_cfg: dict) -> str:
        """
        Auto-select BO method by dimensionality, or return user override.

        If optimization.method is not "auto", that explicit user choice is
        returned directly. Otherwise:
          accuracy_priority=true:
            n_params < 7        -> GP
            7 <= n_params < 10  -> TuRBO
            n_params >= 10      -> SAASBO
          accuracy_priority=false:
            n_params < 7        -> GP
            7 <= n_params < 16  -> TuRBO
            n_params >= 16      -> SAASBO
        """
        method = opt_cfg.get("method", "auto")
        if method != "auto":
            if method == "saasbo" and not _SAASBO_AVAILABLE:
                raise ValueError(
                    "SAASBO was explicitly requested but Pyro is unavailable. "
                    "Install ffopt-lammps[saasbo], or choose method auto/turbo."
                )
            return method

        accuracy_priority = opt_cfg.get("accuracy_priority", True)
        saasbo_threshold  = 10 if accuracy_priority else 16
        turbo_threshold   = 7

        if n_params < turbo_threshold:
            return "gp"
        elif n_params < saasbo_threshold:
            return "turbo"
        else:
            if _SAASBO_AVAILABLE:
                return "saasbo"
            print(f"  WARNING: n_params={n_params} >= {saasbo_threshold} suggests SAASBO "
                  "but pyro is not installed. Using TuRBO instead.")
            return "turbo"

    # ====================================================================== #
    # Main loop                                                              #
    # ====================================================================== #

    def _parse_seed_params(self, path: str):
        """Parse a best_parameters.txt-style file -> {param: value} for warm-start.
        Returns None unless every free BO parameter is present."""
        vals = {}
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in self.param_names:
                    try:
                        vals[k] = float(v.strip())
                    except ValueError:
                        pass
        missing = [p for p in self.param_names if p not in vals]
        if missing:
            print(f"  WARNING: seed file missing {len(missing)} free params "
                  f"(e.g. {missing[:3]}); ignoring seed")
            return None
        return vals

    def _init_seed_from_config(self):
        """Build a warm-start point from per-param 'init' values in atom_types.
        Returns a dict over all free params (init where given, midpoint otherwise),
        or None if no 'init' is specified anywhere."""
        seed = {}
        any_init = False
        for at in self.config["atom_types"]:
            label = at["label"]
            for pname, spec in at["params"].items():
                key = f"{label}_{pname}"
                if key in self.param_names and isinstance(spec, dict) and "init" in spec:
                    seed[key] = float(spec["init"])
                    any_init = True
        if not any_init:
            return None
        for (name, lo, hi) in self.param_space:   # fill the rest with midpoints
            seed.setdefault(name, 0.5 * (lo + hi))
        return seed

    def _resolve_warm_start(self):
        """Return the optional warm-start point and its user-facing source."""
        seed = self._init_seed_from_config()
        if seed is not None:
            return seed, "atom_types init values"

        seed_path = self.config["optimization"].get("seed_params")
        if seed_path and os.path.exists(seed_path):
            return self._parse_seed_params(seed_path), seed_path
        return None, None

    def run(self) -> dict:
        start_time = time.time()
        self._print_header()

        # Phase 1: Initial Latin Hypercube Sampling
        # Skipped on --resume: a loaded checkpoint already holds the initial
        # (and any Phase-2) evaluations, so redoing Phase 1 would duplicate them.
        if not self.all_results:
            print(f"\n{'='*60}")
            print(f"Phase 1: Initial sampling ({self.n_initial} points, LHS)")
            print(f"{'='*60}")

            initial_points = self._generate_lhs(self.n_initial)
            # TuRBO
            # basin instead of a cold, sparse LHS. Priority:
            #   (1) per-param 'init' values in atom_types  (general, material-agnostic)
            #   (2) optimization.seed_params file (legacy)
            seed, src = self._resolve_warm_start()
            if seed:
                initial_points = [seed] + initial_points
                print(f"  Warm-start: seeded initial batch from {src}")
            self._evaluate_and_record(initial_points, "initial")

            if self._best_valid_obj < float("inf"):
                self.best_history.append(self._best_valid_obj)
                n_f = sum(self.all_feasible)
                n_t = len(self.all_feasible)
                print(f"  Best after initial: {self._best_valid_obj:.6f}")
                print(f"  Feasibility: {n_f}/{n_t} ({n_f/n_t*100:.0f}%)")
                self._report_best_progress("initial")
            else:
                print("  WARNING: No successful evaluations in initial phase!")
                self.best_history.append(9999.0)

            # Checkpoint right after the (expensive) initial sampling so a later
            # Phase-2 crash / wall-time hit can --resume without redoing Phase 1.
            if self.ckpt_enabled:
                self.current_round = 0
                self._save_checkpoint()
        else:
            print(f"\n{'='*60}")
            print("Resumed from checkpoint: skipping Phase 1")
            print(f"  {len(self.all_results)} evaluations loaded, "
                  f"best={self._best_valid_obj:.6f}, resuming at round "
                  f"{self.current_round + 1}")
            print(f"{'='*60}")
            self._report_best_progress("resume")

        if self.bo_finished:
            print("\nBO already marked complete in checkpoint; "
                  "continuing with stability audit.")
        else:
            # TuRBO
            self._init_turbo_state()

            # Phase 2: BO-guided acquisition
            print(f"\n{'='*60}")
            print(f"Phase 2: BO acquisition ({self.n_bo_iters} rounds, "
                  f"method={self.bo_method.upper()}, batch={self.batch_size})")
            print(f"{'='*60}")

            no_improve_count = self._infer_no_improve_count()
            prev_best = self._best_valid_obj
            if no_improve_count:
                print(f"  Early-stop state restored: {no_improve_count}/"
                      f"{self.patience} consecutive rounds without sufficient "
                      "improvement")

            # On a fresh run current_round==0, start at round 1. On --resume it is
            # the last completed round, so continue from the next one.
            start_round = self.current_round
            for round_idx in range(start_round, self.n_bo_iters):
                self.current_round = round_idx + 1
                round_start = time.time()
                is_explore  = (self.current_round % self.explore_every == 0)

                tag = "[EXPLORE]" if is_explore else ""
                print(f"\n--- Round {self.current_round}/{self.n_bo_iters} "
                      f"[{self.bo_method.upper()}] {tag} ---")

                if is_explore:
                    candidates = self._propose_exploration_batch()
                else:
                    candidates = self._propose_batch()

                round_label = f"round_{self.current_round}"
                self._evaluate_and_record(candidates, round_label)

                # Track improvement for early stopping
                best_obj = self._best_valid_obj
                self.best_history.append(best_obj)
                round_time = time.time() - round_start
                self.round_times.append(round_time)

                round_improved = best_obj < prev_best - 1.0e-12
                improvement = (prev_best - best_obj) / (abs(prev_best) + 1e-10)
                if improvement > self.min_improvement:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                prev_best = best_obj

                total_evals = len(self.all_results)
                valid_evals = sum(1 for r in self.all_results if r["success"])
                print(f"  Best: {best_obj:.6f} | "
                      f"Evals: {total_evals} ({valid_evals} valid) | "
                      f"Time: {round_time:.1f}s")
                if self.current_round % self.progress_table_every == 0:
                    self._report_best_progress(
                        f"round_{self.current_round}",
                        round_time=round_time,
                        improved=round_improved,
                    )

                # TuRBO
                if (self.bo_method == "turbo" and
                        self._turbo_state is not None and
                        self._turbo_state.restart_triggered):
                    print("  TuRBO restart triggered; resetting trust region state.")
                    self._init_turbo_state()

                # Checkpoint
                if self.ckpt_enabled and self.current_round % self.ckpt_every == 0:
                    self._save_checkpoint()

                # Early stopping
                if self.early_stop and no_improve_count >= self.patience:
                    print(f"\n  Early stopping: no improvement for "
                          f"{self.patience} consecutive rounds")
                    break

            self.bo_finished = True
            if self.ckpt_enabled:
                self._save_checkpoint()

        elapsed = time.time() - start_time
        self._run_stability_audit_if_enabled()
        return self._generate_summary(elapsed)

    def _infer_no_improve_count(self) -> int:
        """Rebuild the early-stop counter from checkpointed best history."""
        count = 0
        for previous, current in zip(self.best_history, self.best_history[1:]):
            improvement = ((previous - current) /
                           (abs(previous) + 1.0e-10))
            if improvement > self.min_improvement:
                count = 0
            else:
                count += 1
        return count

    # ====================================================================== #
    # Acquisition proposal                                                    #
    # ====================================================================== #

    def _propose_batch(self) -> List[Dict[str, float]]:
        """Dispatch to the appropriate BO method."""
        if len(self.train_X) < 3:
            print("  Too few valid points; falling back to LHS")
            return self._generate_lhs(self.batch_size)

        # Active Pareto BO: switch from qLogNEI to qNEHVI when conditions met
        if self.pareto_active and len(self.train_Y2) >= 3:
            return self._propose_batch_pareto()

        if self.bo_method == "gp":
            return self._propose_batch_gp()
        elif self.bo_method == "turbo":
            return self._propose_batch_turbo()
        elif self.bo_method == "saasbo":
            return self._propose_batch_saasbo()
        else:
            raise ValueError(f"Unknown bo_method: {self.bo_method}")

    # Active Pareto BO (GP + compute_surface)

    def _propose_batch_pareto(self) -> List[Dict[str, float]]:
        """
        qNoisyExpectedHypervolumeImprovement over 2 objectives:
          [obj_structural, obj_surface]  (both minimized, negated for BoTorch).

        Uses independent SingleTaskGPs per objective (ModelListGP).
    # GP-BO
        """
        X_norm = normalize(self.train_X, self.bounds)
        Y2     = self.train_Y2   # (N, 2) negated objectives

        try:
            gp_s = SingleTaskGP(X_norm, Y2[:, 0:1],
                                 outcome_transform=Standardize(m=1))
            gp_e = SingleTaskGP(X_norm, Y2[:, 1:2],
                                 outcome_transform=Standardize(m=1))
            model = ModelListGP(gp_s, gp_e)
            mll   = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except Exception:
        # GP-BO
            return self._propose_batch_gp()

        try:
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.pareto_ref_point.tolist(),
                X_baseline=X_norm,
                prune_baseline=True,
            )
            unit_bounds = torch.stack([
                torch.zeros(self.n_params, dtype=torch.double),
                torch.ones( self.n_params, dtype=torch.double),
            ])
            cands_norm, _ = optimize_acqf(
                acq_function=acq, bounds=unit_bounds,
                q=self.batch_size, num_restarts=10, raw_samples=512,
            )
        except Exception:
        # GP-BO
            return self._propose_batch_gp()

        cands = unnormalize(cands_norm, self.bounds)
        cands = self._filter_by_feasibility(cands)
        return self._tensor_to_param_dicts(cands)

    def _compute_pareto_front(self) -> List[dict]:
        """
        Extract the Pareto front from all valid evaluations (post-hoc).

        Returns list of result dicts on the non-dominated front, sorted by
        obj_structural ascending (structural accuracy priority).
        The 'knee' (closest to origin in normalised 2D space) is flagged.
        """
        if not self.pareto_posthoc:
            return []

        valid = [
            r for r in self.all_results
            if r.get("success")
            and np.isfinite(r.get("obj_structural", float("nan")))
            and np.isfinite(r.get("obj_surface",    float("nan")))
        ]
        if len(valid) < 2:
            return valid

        Y = torch.tensor(
            [[r["obj_structural"], r["obj_surface"]] for r in valid],
            dtype=torch.double,
        )
        if _PARETO_AVAILABLE:
            # is_non_dominated expects maximization, so negate both objectives.
            mask = is_non_dominated(-Y)
        else:
            values = Y.numpy()
            mask = torch.tensor([
                not any(
                    np.all(other <= candidate) and np.any(other < candidate)
                    for other in values
                )
                for candidate in values
            ])
        front = [r for r, m in zip(valid, mask.tolist()) if m]
        front.sort(key=lambda r: r["obj_structural"])

        # Flag knee point: minimum distance to origin in normalised space
        if front:
            S_vals = np.array([r["obj_structural"] for r in front])
            E_vals = np.array([r["obj_surface"]    for r in front])
            S_norm = S_vals / (S_vals.max() + 1e-10)
            E_norm = E_vals / (E_vals.max() + 1e-10)
            knee_idx = int(np.argmin(S_norm**2 + E_norm**2))
            for i, r in enumerate(front):
                r["pareto_knee"] = (i == knee_idx)

        return front

        # GP-BO

    def _propose_batch_gp(self) -> List[Dict[str, float]]:
        """Standard GP + qLogNEI acquisition (BoTorch)."""
        gp_X, gp_Y = self._get_gp_training_data()

        X_norm = normalize(gp_X, self.bounds)
        Y_neg  = -gp_Y   # BoTorch maximises; we minimise objective

        model = SingleTaskGP(X_norm, Y_neg, outcome_transform=Standardize(m=1))
        mll   = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  Warning: GP fitting issue ({e}); using LHS fallback")
            return self._generate_lhs(self.batch_size)

        X_norm_full = normalize(self.train_X, self.bounds)
        acq = qLogNoisyExpectedImprovement(model=model, X_baseline=X_norm_full)

        unit_bounds = torch.stack([
            torch.zeros(self.n_params, dtype=torch.double),
            torch.ones( self.n_params, dtype=torch.double),
        ])

        try:
            cands_norm, _ = optimize_acqf(
                acq_function=acq, bounds=unit_bounds,
                q=self.batch_size, num_restarts=10, raw_samples=512,
            )
        except Exception as e:
            print(f"  Warning: acquisition optimisation failed ({e})")
            return self._generate_lhs(self.batch_size)

        cands = unnormalize(cands_norm, self.bounds)
        cands = self._filter_by_feasibility(cands)
        return self._tensor_to_param_dicts(cands)

    # TuRBO

    def _init_turbo_state(self):
        """Initialise (or restart) the TuRBO trust-region state."""
        self._turbo_state = TurboState(
            dim=self.n_params,
            batch_size=self.batch_size,
            length=self.turbo_length_init,
            length_min=self.turbo_length_min,
            length_max=self.turbo_length_max,
        )
        if self._best_valid_obj < float("inf"):
            self._turbo_state.best_value = self._best_valid_obj

    def _propose_batch_turbo(self) -> List[Dict[str, float]]:
        """
        TuRBO: Thompson Sampling within an axis-aligned trust region
        centred at the current best point.

        Reference: Eriksson et al. (2019) "Scalable Global Optimization
        via Local Bayesian Optimization", NeurIPS.
        """
        state = self._turbo_state

        # Centre trust region on current best point (in normalised space)
        best_idx   = self.train_Y.argmin()
        X_best_raw = self.train_X[best_idx]
        X_best_norm = normalize(X_best_raw.unsqueeze(0), self.bounds).squeeze(0)

        # Axis-aligned trust region: half-side = length/2 in [0,1]
        half = state.length / 2.0
        lb   = torch.clamp(X_best_norm - half, 0.0, 1.0)
        ub   = torch.clamp(X_best_norm + half, 0.0, 1.0)

        # Train GP on normalised data
        gp_X, gp_Y = self._get_gp_training_data()
        X_norm = normalize(gp_X, self.bounds)
        Y_neg  = -gp_Y

        model = SingleTaskGP(X_norm, Y_neg, outcome_transform=Standardize(m=1))
        mll   = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  Warning: TuRBO GP fitting issue ({e}); using LHS in TR")
            # Sample randomly within trust region
            X_rand_norm = lb + (ub - lb) * torch.rand(
                self.batch_size, self.n_params, dtype=torch.double)
            cands = unnormalize(X_rand_norm, self.bounds)
            return self._tensor_to_param_dicts(cands)

        # Thompson Sampling: draw self.batch_size candidates from trust region
        n_cands = min(5000, max(2000, 200 * self.n_params))
        X_cand_norm = lb + (ub - lb) * torch.rand(
            n_cands, self.n_params, dtype=torch.double)

        model.eval()
        ts = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            # MaxPosteriorSampling selects batch_size points from candidates
            X_next_norm = ts(X_cand_norm, num_samples=self.batch_size)

        cands = unnormalize(X_next_norm, self.bounds)

        # Update trust region state with objectives from the LAST batch
        # (we use the best_history delta as a proxy)
        if len(self.best_history) >= 2:
            recent_objs = [r["objective"] for r in self.all_results
                           if r.get("round") == self.current_round - 1]
            if recent_objs:
                self._turbo_state = _update_turbo_state(state, recent_objs)

        print(f"  TuRBO: TR length={state.length:.4f}  "
              f"fail={state.failure_counter}/{state.failure_tolerance}  "
              f"succ={state.success_counter}/{state.success_tolerance}")

        return self._tensor_to_param_dicts(cands)

    # SAASBO

    def _propose_batch_saasbo(self) -> List[Dict[str, float]]:
        """
        SAASBO: Sparse Axis-Aligned Subspace BO.
        Uses SaasFullyBayesianSingleTaskGP which marginalises over kernel
        hyperparameters via NUTS, imposing a sparsity prior that concentrates
        on the most sensitive dimensions.

        Reference: Eriksson & Jankowiak (2021) "High-Dimensional Bayesian
        Optimization with Sparse Axis-Aligned Subspaces", UAI.
        """
        if not _SAASBO_AVAILABLE:
            print("  SAASBO not available (pyro missing); falling back to TuRBO")
            self.bo_method = "turbo"
            self._init_turbo_state()
            return self._propose_batch_turbo()

        gp_X, gp_Y = self._get_gp_training_data()
        X_norm = normalize(gp_X, self.bounds)
        # botorch needs train_Y shaped (n, 1). gp_Y is already (n, 1) in this
        # pipeline; guard for the 1-D case too. (BoTorch maximises -> negate.)
        Y = gp_Y if gp_Y.dim() == 2 else gp_Y.unsqueeze(-1)
        Y_neg  = -Y

        model = SaasFullyBayesianSingleTaskGP(
            train_X=X_norm, train_Y=Y_neg,
            train_Yvar=torch.full_like(Y_neg, 1e-6),
        )
        try:
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=self.saasbo_warmup_steps,
                num_samples=self.saasbo_num_samples,
                thinning=16,
                disable_progbar=True,
            )
        except Exception as e:
            print(f"  Warning: SAASBO NUTS sampling failed ({e}); "
                  "falling back to GP-BO.")
            return self._propose_batch_gp()

        X_norm_full = normalize(self.train_X, self.bounds)
        acq = qLogNoisyExpectedImprovement(model=model, X_baseline=X_norm_full)

        unit_bounds = torch.stack([
            torch.zeros(self.n_params, dtype=torch.double),
            torch.ones( self.n_params, dtype=torch.double),
        ])

        try:
            cands_norm, _ = optimize_acqf(
                acq_function=acq, bounds=unit_bounds,
                q=self.batch_size, num_restarts=10,
                raw_samples=min(1024, 64 * self.n_params),
            )
        except Exception as e:
            print(f"  Warning: SAASBO acquisition optimisation failed ({e})")
            return self._generate_lhs(self.batch_size)

        cands = unnormalize(cands_norm, self.bounds)
        return self._tensor_to_param_dicts(cands)

    # ====================================================================== #
    # Feasibility filtering                                                   #
    # ====================================================================== #

    def _filter_by_feasibility(self, candidates: Tensor) -> Tensor:
        """Replace low-probability-feasible candidates with better alternatives."""
        if not self.use_feasibility_classifier or self._feasibility_model is None:
            return candidates
        if len(self.all_X) < 20:
            return candidates

        cand_np = normalize(candidates, self.bounds).numpy()
        try:
            probs = self._feasibility_model.predict_proba(cand_np)[:, 1]
        except Exception:
            return candidates

        n_replaced = 0
        for i in range(len(candidates)):
            if probs[i] < 0.3:
                replacement = self._sample_feasible_point()
                if replacement is not None:
                    candidates[i] = replacement
                    n_replaced += 1

        if n_replaced > 0:
            print(f"  Feasibility filter: replaced {n_replaced}/{len(candidates)} "
                  "candidates")
        return candidates

    def _sample_feasible_point(self) -> Optional[Tensor]:
        if self._feasibility_model is None:
            return None
        for _ in range(100):
            x = torch.rand(self.n_params, dtype=torch.double)
            try:
                prob = self._feasibility_model.predict_proba(
                    x.unsqueeze(0).numpy())[0, 1]
                if prob > 0.5:
                    return unnormalize(x.unsqueeze(0), self.bounds).squeeze(0)
            except Exception:
                pass
        return None

    def _update_feasibility_classifier(self):
        """Train GP classifier (feasible vs infeasible) on all data so far."""
        if len(self.all_X) < 10:
            return
        n_f   = sum(self.all_feasible)
        n_inf = len(self.all_feasible) - n_f
        if n_f < 3 or n_inf < 3:
            return

        X_np = normalize(self.all_X, self.bounds).numpy()
        y_np = np.array(self.all_feasible, dtype=float)
        try:
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)
            self._feasibility_model = Pipeline([
                ("scaler", StandardScaler()),
                ("gpc",    GaussianProcessClassifier(
                               kernel=kernel,
                               random_state=self.seed,
                               max_iter_predict=100,
                               warm_start=True)),
            ])
            self._feasibility_model.fit(X_np, y_np)
        except Exception as e:
            print(f"  Feasibility classifier warning: {e}")
            self._feasibility_model = None

    # ====================================================================== #
    # Exploration                                                             #
    # ====================================================================== #

    def _propose_exploration_batch(self) -> List[Dict[str, float]]:
        """Periodic exploration: LHS + GP-uncertainty sampling."""
        candidates = []
        candidates.extend(self._generate_lhs(self.explore_n_random))
        if len(self.train_X) >= 3 and self.explore_n_uncertain > 0:
            candidates.extend(
                self._propose_uncertain_points(self.explore_n_uncertain))
        else:
            candidates.extend(self._generate_lhs(self.explore_n_uncertain))
        return candidates

    def _propose_uncertain_points(self, n_points: int) -> List[Dict[str, float]]:
        """Return points with highest GP posterior standard deviation."""
        X_norm = normalize(self.train_X, self.bounds)
        Y_neg  = -self.train_Y

        model = SingleTaskGP(X_norm, Y_neg, outcome_transform=Standardize(m=1))
        mll   = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            return self._generate_lhs(n_points)

        n_cand = 1000
        X_cand = torch.rand(n_cand, self.n_params, dtype=torch.double)

        model.eval()
        with torch.no_grad():
            std = model.posterior(X_cand).variance.sqrt().squeeze()

        scores = std.numpy()
        if self._feasibility_model is not None:
            try:
                probs  = self._feasibility_model.predict_proba(X_cand.numpy())[:, 1]
                scores = scores * probs
            except Exception:
                pass

        top_idx = np.argsort(-scores)[:n_points]
        X_top   = unnormalize(X_cand[top_idx], self.bounds)
        return self._tensor_to_param_dicts(X_top)

    # ====================================================================== #
    # Evaluation                                                              #
    # ====================================================================== #

    def _evaluate_and_record(self,
                             param_dicts: List[Dict[str, float]],
                             label: str):
        """Run LAMMPS batch, record all results, update GP training data."""
        eval_dir = os.path.join(self.work_dir, label)
        os.makedirs(eval_dir, exist_ok=True)

        results = self.runner.evaluate_batch(param_dicts, eval_dir)

        new_X, new_Y, new_Y2, new_all_X, new_feasible = [], [], [], [], []

        for result in results:
            # Build CSV row
            entry: Dict = {
                "round":      self.current_round,
                "label":      label,
                "success":    result.success,
                "objective":  result.objective,
                "error_msg":  result.error_msg,
            }
            entry.update(objective_provenance(active_targets(self.config)))
            # All free BO params
            for name in self.param_names:
                entry[name] = result.params.get(name, float("nan"))
            # Per-target computed values and errors
            for prop in self.config["targets"]:
                entry[f"calc_{prop}"]  = result.properties.get(prop,  float("nan"))
                entry[f"error_{prop}"] = result.per_property_error.get(prop, float("nan"))
            # Surface energy decomposition (auxiliary data for NN surrogate)
            for aux in ("E_complete", "E_split", "A_xy"):
                entry[f"calc_{aux}"] = result.properties.get(aux, float("nan"))

            # Pareto objective columns
            entry["obj_structural"] = result.obj_structural
            entry["obj_surface"]    = result.obj_surface

            self.all_results.append(entry)

            x = [result.params.get(name, float("nan")) for name in self.param_names]
            new_all_X.append(x)
            new_feasible.append(result.success)

            if result.success and np.isfinite(result.objective):
                new_X.append(x)
                new_Y.append([result.objective])
                if result.objective < self._best_valid_obj:
                    self._best_valid_obj = result.objective
                # Pareto training: negate because BoTorch maximizes
                if (self.pareto_active
                        and np.isfinite(result.obj_structural)
                        and np.isfinite(result.obj_surface)):
                    new_Y2.append([-result.obj_structural, -result.obj_surface])
            else:
                cap = self._compute_penalty_cap()
                new_X.append(x)
                new_Y.append([cap])

        if new_X:
            self.train_X = torch.cat(
                [self.train_X, torch.tensor(new_X, dtype=torch.double)], dim=0)
            self.train_Y = torch.cat(
                [self.train_Y, torch.tensor(new_Y, dtype=torch.double)], dim=0)
        if new_Y2:
            self.train_Y2 = torch.cat(
                [self.train_Y2, torch.tensor(new_Y2, dtype=torch.double)], dim=0)

        if new_all_X:
            self.all_X = torch.cat(
                [self.all_X, torch.tensor(new_all_X, dtype=torch.double)], dim=0)
            self.all_feasible.extend(new_feasible)

        self._update_feasibility_classifier()

        n_ok = sum(1 for r in results if r.success)
        print(f"  Batch '{label}': {n_ok}/{len(results)} successful")

    def _compute_penalty_cap(self) -> float:
        """Soft cap: penalty_cap_multiplier * worst valid objective (>=100)."""
        valid_objs = [r["objective"] for r in self.all_results if r["success"]]
        if not valid_objs:
            return 10.0
        return min(max(valid_objs) * self.penalty_cap_multiplier, 100.0)

    def _run_stability_audit_if_enabled(self) -> None:
        """
        Re-evaluate top BO candidates with multiple NPT seeds and write:
          - stability_replicates.csv: one row per candidate x seed
          - stable_results.csv: one mean/std row per candidate, all_results-like

        The NN loader prefers stable_results.csv when present. This makes the
        surrogate learn the reproducible parameter-to-property map instead of
        single-seed thermal noise.
        """
        if (not self.stability_enabled or self.stability_top_k <= 0
                or not self.stability_seeds):
            return

        valid = [r for r in self.all_results
                 if r.get("success") and np.isfinite(r.get("objective", np.nan))]
        if not valid:
            print("\nStability audit: no successful candidates; skipping.")
            return

        valid = sorted(valid, key=lambda r: r["objective"])[:self.stability_top_k]
        audit_dir = os.path.join(self.work_dir, "stability_audit")
        os.makedirs(audit_dir, exist_ok=True)
        print(f"\nStability audit: top {len(valid)} candidates x "
              f"{len(self.stability_seeds)} seeds")

        active_props = [
            p for p, info in self.config["targets"].items()
            if not (p == "surf_energy" and not self.config["lammps"]["compute_surface"])
            and not (p == "ead" and not self.config.get("adsorption", {}).get("enabled", False))
            and float(info.get("weight", 1.0)) > 0.0
        ]

        replicate_path = os.path.join(self.work_dir, "stability_replicates.csv")
        stable_path = os.path.join(self.work_dir, "stable_results.csv")
        replicate_rows = self._read_csv_if_exists(replicate_path)
        stable_rows = self._read_csv_if_exists(stable_path)
        completed_ranks = {
            int(r["candidate_rank"])
            for r in stable_rows
            if str(r.get("candidate_rank", "")).isdigit()
        }
        if completed_ranks:
            print(f"  Resuming audit: {len(completed_ranks)} candidates "
                  "already summarized; skipping them.")

        pending = []
        for rank, row in enumerate(valid, start=1):
            if rank in completed_ranks:
                print(f"  audit #{rank:02d}: already complete; skipping")
            else:
                pending.append((rank, row))

        if not pending:
            print("  Stability audit: all requested candidates already complete.")
        else:
            self._prepare_stability_audit_references(audit_dir)
            audit_workers = max(1, min(self.stability_max_workers, len(pending)))
            print(f"  Audit parallelism: {audit_workers} candidate workers "
                  f"(each candidate runs {len(self.stability_seeds)} seeds)")

            futures = {}
            with ThreadPoolExecutor(max_workers=audit_workers) as executor:
                for rank, row in pending:
                    fut = executor.submit(
                        self._audit_one_candidate,
                        rank, row, active_props, audit_dir)
                    futures[fut] = rank

                for fut in as_completed(futures):
                    rank = futures[fut]
                    try:
                        rep_new, agg, msg = fut.result()
                    except Exception as e:
                        print(f"  audit #{rank:02d}: failed with exception: {e}")
                        continue

                    # If this rank was partially written by a previous interrupted
                    # run, replace its rows atomically after the full candidate is
                    # complete.
                    replicate_rows = [
                        r for r in replicate_rows
                        if str(r.get("candidate_rank")) != str(rank)
                    ]
                    stable_rows = [
                        r for r in stable_rows
                        if str(r.get("candidate_rank")) != str(rank)
                    ]
                    replicate_rows.extend(rep_new)
                    stable_rows.append(agg)
                    print(msg)

                    self._write_csv(replicate_path, replicate_rows)
                    self._write_csv(stable_path, stable_rows)

        self._write_csv(replicate_path, replicate_rows)
        self._write_csv(stable_path, stable_rows)
        n_stable = sum(1 for r in stable_rows if self._csv_truthy(r["success"]))
        print(f"\nStability audit complete: {n_stable}/{len(stable_rows)} "
              f"candidates passed. NN will prefer "
              f"{self.work_dir}/stable_results.csv")

    def _prepare_stability_audit_references(self, audit_dir: str) -> None:
        """Compute parameter-independent references before threaded audit work."""
        if getattr(self.runner, "ads_enabled", False) and self.runner._slab_pe is None:
            self.runner._slab_pe = self.runner._run_ads_box(
                {}, audit_dir, "_slab_ref", self.runner.ads_slab,
                self.runner.ads_slab_input, has_charge=False, save_traj=False)
            print(f"  [adsorption] reference E_slab = "
                  f"{self.runner._slab_pe} kcal/mol")

    def _audit_one_candidate(self,
                             rank: int,
                             row: dict,
                             active_props: List[str],
                             audit_dir: str) -> Tuple[List[dict], dict, str]:
        """Run all stability seeds for one candidate and return CSV rows."""
        params = {name: float(row[name]) for name in self.param_names}
        cand_dir = os.path.join(audit_dir, f"candidate_{rank:03d}")
        results = self.runner.evaluate_replicates(
            params, cand_dir, self.stability_seeds, save_traj=False)

        replicate_rows = []
        for seed, result in zip(self.stability_seeds, results):
            rep = {
                "candidate_rank": rank,
                "seed": seed,
                "success": result.success,
                "objective": result.objective,
                "error_msg": result.error_msg,
            }
            rep.update(objective_provenance(active_targets(self.config)))
            rep.update(params)
            for prop in active_props:
                rep[f"calc_{prop}"] = result.properties.get(prop, np.nan)
                rep[f"error_{prop}"] = result.per_property_error.get(prop, np.nan)
            replicate_rows.append(rep)

        ok = [r for r in results if r.success and np.isfinite(r.objective)]
        failure_rate = 1.0 - len(ok) / max(len(results), 1)
        objectives = np.array([r.objective for r in ok], dtype=float)

        mean_props = {}
        prop_rel_stds = []
        for prop in active_props:
            vals = np.array([r.properties.get(prop, np.nan) for r in ok],
                            dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                mean_props[prop] = float(vals.mean())
                target = abs(float(self.config["targets"][prop]["value"]))
                prop_rel_stds.append(float(vals.std(ddof=0) / (target + 1e-10)))

        if mean_props and len(mean_props) == len(active_props):
            mean_objective, per_prop = self.runner._compute_objective(mean_props)
        else:
            mean_objective, per_prop = LARGE_PENALTY, {}

        objective_std = (
            float(objectives.std(ddof=0)) if objectives.size else float("inf"))
        max_prop_rel_std = max(prop_rel_stds) if prop_rel_stds else float("inf")
        stable = (
            failure_rate == 0.0
            and objective_std <= self.stability_max_obj_std
            and max_prop_rel_std <= self.stability_max_prop_rel_std
            and np.isfinite(mean_objective)
        )
        robust_objective = (
            mean_objective
            + self.stability_noise_penalty * objective_std
            + self.stability_failure_penalty * failure_rate
        )

        agg = {
            "round": row.get("round", self.current_round),
            "label": "stability_audit",
            "candidate_rank": rank,
            "success": stable,
            "objective": robust_objective,
            "objective_mean": mean_objective,
            "objective_std": objective_std,
            "failure_rate": failure_rate,
            "max_property_rel_std": max_prop_rel_std,
        }
        agg.update(objective_provenance(active_targets(self.config)))
        agg.update(params)
        for prop in active_props:
            agg[f"calc_{prop}"] = mean_props.get(prop, np.nan)
            agg[f"error_{prop}"] = per_prop.get(prop, np.nan)

        msg = (f"  audit #{rank:02d}: robust={robust_objective:.6f} "
               f"mean={mean_objective:.6f} std={objective_std:.6f} "
               f"fail={failure_rate:.2f} stable={stable}")
        return replicate_rows, agg, msg

    @staticmethod
    def _read_csv_if_exists(path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _csv_truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y")

    @staticmethod
    def _write_csv(path: str, rows: List[dict]) -> None:
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _get_gp_training_data(self) -> Tuple[Tensor, Tensor]:
        """
        Return (X, Y) for GP fitting, capped at MAX_GP_POINTS best points.
        Sorted selection keeps the GP focused on the promising region.
        """
        if len(self.train_X) <= MAX_GP_POINTS:
            return self.train_X, self.train_Y
        idx = torch.argsort(self.train_Y.squeeze())[:MAX_GP_POINTS]
        return self.train_X[idx], self.train_Y[idx]

    # ====================================================================== #
    # LHS sampling                                                            #
    # ====================================================================== #

    def _generate_lhs(self, n_points: int) -> List[Dict[str, float]]:
        """Latin Hypercube samples over the full parameter space."""
        sampler = qmc.LatinHypercube(
            d=self.n_params,
            seed=self.seed + self.current_round * 1000,
        )
        unit = sampler.random(n=n_points)
        param_dicts = []
        for i in range(n_points):
            d = {}
            for j, (name, lo, hi) in enumerate(self.param_space):
                d[name] = lo + unit[i, j] * (hi - lo)
            param_dicts.append(d)
        return param_dicts

    # ====================================================================== #
    # Utility                                                                 #
    # ====================================================================== #

    def _tensor_to_param_dicts(self, x: Tensor) -> List[Dict[str, float]]:
        """Convert (N x n_params) tensor to list of param name/value dicts."""
        out = []
        for i in range(x.shape[0]):
            d = {name: x[i, j].item() for j, name in enumerate(self.param_names)}
            out.append(d)
        return out

    # ====================================================================== #
    # Checkpoint                                                              #
    # ====================================================================== #

    def _save_checkpoint(self):
        ckpt = {
            "round":          self.current_round,
            "bo_method":      self.bo_method,
            "train_X":        self.train_X.numpy().tolist(),
            "train_Y":        self.train_Y.numpy().tolist(),
            "all_X":          self.all_X.numpy().tolist(),
            "all_feasible":   self.all_feasible,
            "all_results":    self.all_results,
            "best_history":   self.best_history,
            "best_valid_obj": self._best_valid_obj,
            "round_times":    self.round_times,
            "bo_finished":    self.bo_finished,
            "param_names":    self.param_names,
            "timestamp":      datetime.now().isoformat(),
        }
        # TuRBO
        if self._turbo_state is not None:
            ts = self._turbo_state
            ckpt["turbo_state"] = {
                "length":          ts.length,
                "failure_counter": ts.failure_counter,
                "success_counter": ts.success_counter,
                "best_value":      ts.best_value,
            }

        path   = os.path.join(self.ckpt_dir,
                              f"checkpoint_round_{self.current_round}.json")
        latest = os.path.join(self.ckpt_dir, "latest.json")
        for p in (path, latest):
            with open(p, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

    def load_checkpoint(self, path: str = None) -> bool:
        if path is None:
            # Each run creates a fresh timestamped work_dir, so this run's own
            # ckpt_dir is empty. For --resume to actually continue, auto-discover
            # the most recent latest.json among ALL prior bo_<system>_* runs.
            own = os.path.join(self.ckpt_dir, "latest.json")
            if os.path.exists(own):
                path = own
            else:
                import glob
                ckpt_sub = os.path.basename(self.ckpt_dir)
                sysname  = self.config["manifest"]["system_name"]
                patterns = [os.path.join(
                    self.run_root,
                    f"bo_{sysname}_*",
                    ckpt_sub,
                    "latest.json",
                )]
                if os.path.abspath(self.run_root) != os.path.abspath("."):
                    patterns.append(os.path.join(
                        f"bo_{sysname}_*", ckpt_sub, "latest.json"
                    ))
                cands = [
                    p for pattern in patterns for p in glob.glob(pattern)
                    if os.path.abspath(os.path.dirname(os.path.dirname(p)))
                       != os.path.abspath(self.work_dir)
                ]
                if not cands:
                    print(f"No checkpoint found (searched {patterns})")
                    return False
                # Pick the MOST PROGRESSED checkpoint (highest round), not just
            # the newest file; a fresh run round-0 checkpoint can have a
                # newer mtime than an earlier run that reached round 60.
                def _ckpt_round(p):
                    try:
                        with open(p) as fh:
                            return json.load(fh).get("round", -1)
                    except Exception:
                        return -1
                path = max(cands, key=lambda p: (_ckpt_round(p),
                                                 os.path.getmtime(p)))
                print(f"Auto-resuming from checkpoint: {path} "
                      f"(round {_ckpt_round(path)})")
        if not os.path.exists(path):
            print(f"No checkpoint at {path}")
            return False

        with open(path) as f:
            ckpt = json.load(f)

        self.current_round    = ckpt["round"]
        self.train_X          = torch.tensor(ckpt["train_X"],  dtype=torch.double)
        self.train_Y          = torch.tensor(ckpt["train_Y"],  dtype=torch.double)
        self.all_X            = torch.tensor(ckpt["all_X"],    dtype=torch.double)
        self.all_feasible     = ckpt["all_feasible"]
        self.all_results      = ckpt["all_results"]
        self.best_history     = ckpt["best_history"]
        self._best_valid_obj  = ckpt.get("best_valid_obj", float("inf"))
        self.round_times      = ckpt.get("round_times", [])
        self.bo_finished      = bool(ckpt.get("bo_finished", False))

        # TuRBO
        ts_ckpt = ckpt.get("turbo_state")
        if ts_ckpt and self.bo_method == "turbo":
            self._init_turbo_state()
            self._turbo_state.length          = ts_ckpt["length"]
            self._turbo_state.failure_counter = ts_ckpt["failure_counter"]
            self._turbo_state.success_counter = ts_ckpt["success_counter"]
            self._turbo_state.best_value      = ts_ckpt["best_value"]

        # Verify param_names match (catches config changes between runs)
        ckpt_names = ckpt.get("param_names", [])
        if ckpt_names and ckpt_names != self.param_names:
            print("  WARNING: checkpoint param_names differ from current config!")
            print(f"    checkpoint : {ckpt_names}")
            print(f"    current    : {self.param_names}")

        self._update_feasibility_classifier()
        print(f"Resumed from round {self.current_round}, "
              f"{len(self.all_results)} evaluations")
        return True

    # ====================================================================== #
    # Summary and output files                                                #
    # ====================================================================== #

    def _generate_summary(self, elapsed: float) -> dict:
        valid = [r for r in self.all_results if r["success"]]
        if not valid:
            print("\nNo valid results; cannot generate summary.")
            return {}

        best  = min(valid, key=lambda r: r["objective"])
        n_f   = sum(self.all_feasible)
        n_t   = len(self.all_feasible)

        try:
            best_idx = self.all_results.index(best)
        except ValueError:
            best_idx = -1

        print(f"\n{'='*60}")
        print("BO Complete")
        print(f"{'='*60}")
        print(f"Total time  : {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
        print(f"Evaluations : {len(self.all_results)} total, {len(valid)} valid")
        print(f"Feasibility : {n_f}/{n_t} ({n_f/n_t*100:.0f}%)")
        print(f"BO method   : {self.bo_method.upper()}")
        print(f"Best objective : {best['objective']:.6f}")
        print(f"  Found at round {best.get('round','?')}, "
              f"label '{best.get('label','?')}'")
        print(f"  Evaluation #{best_idx+1} / {len(self.all_results)}")

        print("\nBest parameters:")
        for name, lo, hi in self.param_space:
            val = best.get(name, float("nan"))
            print(f"  {name:<28} = {val:.6f}  [{lo}, {hi}]")

        print("\nTarget comparison:")
        hdr = f"  {'Property':<14} {'Calculated':>12} {'Target':>10} {'Error%':>8}"
        print(hdr)
        print(f"  {'-'*46}")
        for prop, info in self._active_targets.items():
            calc   = best.get(f"calc_{prop}", float("nan"))
            target = float(info["value"])
            err    = best.get(f"error_{prop}", float("nan"))
            flag   = "OK" if err < 5.0 else "!!"
            print(f"  {prop:<14} {calc:>12.4f} {target:>10.4f} {err:>7.2f}%  {flag}")

        # Pareto front (post-hoc)
        pareto_front = self._compute_pareto_front()
        if pareto_front:
            knee = next((r for r in pareto_front if r.get("pareto_knee")), None)
            print(f"\nPareto front: {len(pareto_front)} non-dominated solutions")
            print(f"  {'obj_structural':>16}  {'obj_surface':>12}  {'knee':>5}")
            for r in pareto_front:
                k = "\u2605" if r.get("pareto_knee") else " "
                print(f"  {r['obj_structural']:>16.6f}  "
                      f"{r['obj_surface']:>12.2f}%  {k}")
            if knee:
                print("\nKnee point (auto-selected):")
                for name in self.param_names:
                    print(f"  {name:<28} = {knee.get(name, float('nan')):.6f}")

        self._save_summary_files(best, elapsed, valid, pareto_front)
        return best

    def _save_summary_files(self, best: dict, elapsed: float, valid: list,
                            pareto_front=None):
        import pandas as pd

        # all_results.csv (includes obj_structural + obj_surface columns)
        df       = pd.DataFrame(self.all_results)
        csv_path = os.path.join(self.work_dir, "all_results.csv")
        df.to_csv(csv_path, index=False)

        # best_parameters.txt (read by run.py --dry-run auto-loader)
        param_path = os.path.join(self.work_dir, "best_parameters.txt")
        with open(param_path, "w") as f:
            f.write("# FFOpt-LAMMPS force-field optimizer\n")
            f.write(f"# System    : {self.config['manifest']['system_name']}\n")
            f.write(f"# Method    : {self.bo_method.upper()}\n")
            f.write(f"# Objective : {best['objective']:.6f}\n")
            f.write(f"# Pareto    : active={self.pareto_active} posthoc={self.pareto_posthoc}\n")
            f.write(f"# Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for name, lo, hi in self.param_space:
                val = best.get(name, float("nan"))
                f.write(f"{name} = {val:.8f}  # range [{lo}, {hi}]\n")

        # optimization_report.txt
        n_f = sum(self.all_feasible)
        n_t = len(self.all_feasible)
        report_path = os.path.join(self.work_dir, "optimization_report.txt")
        with open(report_path, "w") as f:
            f.write("FFOpt-LAMMPS Force Field Optimization Report\n")
            f.write(f"System      : {self.config['manifest']['system_name']}\n")
            f.write(f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"BO method   : {self.bo_method.upper()}\n")
            f.write(f"n_params    : {self.n_params}\n")
            f.write(f"accuracy_pr : {self.config['optimization'].get('accuracy_priority', True)}\n")
            f.write(f"Pareto      : active={self.pareto_active}\n")
            f.write(f"Time        : {elapsed/60:.1f} min\n")
            f.write(f"Evaluations : {len(self.all_results)} ({len(valid)} valid)\n")
            f.write(f"Feasibility : {n_f}/{n_t} ({n_f/n_t*100:.0f}%)\n")
            f.write(f"Rounds      : {self.current_round}\n\n")
            f.write(f"Best objective : {best['objective']:.6f}\n\n")
            f.write("Parameters:\n")
            for name, lo, hi in self.param_space:
                val = best.get(name, float("nan"))
                f.write(f"  {name:<28} = {val:.6f}  [{lo}, {hi}]\n")
            f.write("\nProperties:\n")
            for prop, info in self.config["targets"].items():
                if (prop == "surf_energy" and
                        not self.config["lammps"]["compute_surface"]):
                    continue
                calc   = best.get(f"calc_{prop}", float("nan"))
                target = float(info["value"])
                err    = best.get(f"error_{prop}", float("nan"))
                f.write(f"  {prop:<14} calc={calc:.4f}  "
                        f"target={target:.4f}  error={err:.2f}%\n")

        # pareto_front.csv
        pareto_path = os.path.join(self.work_dir, "pareto_front.csv")
        if pareto_front:
            pd.DataFrame(pareto_front).to_csv(pareto_path, index=False)
        else:
            pd.DataFrame().to_csv(pareto_path, index=False)

        print("\nFiles saved:")
        print(f"  {csv_path}")
        print(f"  {param_path}")
        print(f"  {report_path}")
        print(f"  {pareto_path}")

    # ====================================================================== #
    # Header                                                                  #
    # ====================================================================== #

    def _print_header(self):
        print(f"\n{'#'*60}")
        print("# FFOpt-LAMMPS Bayesian Optimizer")
        print(f"# {self.bo_method.upper()} | Pareto active={self.pareto_active}")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        mf = self.config["manifest"]
        print(f"\nSystem      : {mf['system_name']}")
        print(f"Structure   : {mf.get('crystal_structure', 'N/A')}")
        print(f"Facet       : {mf.get('surface_facet', 'N/A')}")
        print(f"Surf energy : {'YES' if self.config['lammps']['compute_surface'] else 'NO (bulk only)'}")
        print(f"Charge      : {'YES (Coulomb)' if self.config['charge']['enabled'] else 'NO (LJ only)'}")

        print(f"\nFree parameters ({self.n_params}):")
        for name, lo, hi in self.param_space:
            print(f"  {name:<28} : [{lo}, {hi}]")

        print(f"\nTargets ({len(self.config['targets'])}):")
        for prop, info in self.config["targets"].items():
            w      = info.get("weight", 1.0)
            active = not (prop == "surf_energy" and
                          not self.config["lammps"]["compute_surface"])
            status = "" if active else "  [DISABLED]"
            print(f"  {prop:<14} = {info['value']}  (weight={w}){status}")

        print("\nBO settings:")
        warm_start_count = int(self._resolve_warm_start()[0] is not None)
        initial_total = self.n_initial + warm_start_count
        search_budget = initial_total + self.n_bo_iters * self.batch_size
        audit_budget = (
            self.stability_top_k * len(self.stability_seeds)
            if self.stability_enabled
            else 0
        )
        print(f"  Method         : {self.bo_method.upper()}")
        print(f"  accuracy_prior : {self.config['optimization'].get('accuracy_priority', True)}")
        print(f"  Pareto active  : {self.pareto_active}")
        print(f"  Initial LHS    : {self.n_initial}")
        print(f"  Warm-start     : +{warm_start_count} centre")
        print(f"  Initial total  : {initial_total}")
        print(f"  BO rounds      : {self.n_bo_iters} x batch {self.batch_size}")
        print(f"  Search evals   : {search_budget}")
        if self.stability_enabled:
            print(
                f"  Stability audit: top {self.stability_top_k} x "
                f"{len(self.stability_seeds)} seeds (max {audit_budget})"
            )
        print(f"  Max stage evals: {search_budget + audit_budget}")
