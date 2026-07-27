#!/usr/bin/env python3
"""
utils/sensitivity.py  |  Parameter Sensitivity Analysis  |  v8

Computes sensitivity of the objective to each free BO parameter using the
trained per-property EnsembleMLP ensemble.

Two methods:
  sobol  : Sobol total-order sensitivity indices (variance-based, most robust)
  grad   : mean |∂obj/∂param| via finite differences over the Sobol pool
            (faster, less rigorous, good for a quick ranking)

Usage:
  python utils/sensitivity.py --config configs/recipes/cluster_full.yaml --nn-dir nn_output_*/
  python utils/sensitivity.py --config configs/recipes/cluster_full.yaml --nn-dir nn_output_*/ --method grad
  python utils/sensitivity.py --config configs/recipes/cluster_full.yaml --nn-dir nn_output_*/ --n-samples 8192

Outputs (in --nn-dir):
  sensitivity_sobol.csv  — Sobol total-order indices per parameter
  sensitivity_grad.csv   — mean |gradient| per parameter
  sensitivity.png        — ranked bar chart (both methods)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config_loader import load_config
from engine.nn_surrogate import load_ensemble_from_file
from engine.optimizer import ForceFieldOptimizer


# ============================================================================
# Objective function over NN ensemble
# ============================================================================

def _build_obj_fn(ensembles, feat_builder, targets_cfg,
                  physics_features: bool, compute_surface: bool):
    """
    Return a vectorised function  f(X) → (N,)  that evaluates the
    weighted RMSE objective using all per-property ensemble means.
    X : np.ndarray shape (N, D), physical units.
    """
    active_targets = {
        p: info for p, info in targets_cfg.items()
        if not (p == "surf_energy" and not compute_surface)
        and float(info.get("weight", 1.0)) > 0.0
    }

    def obj(X: np.ndarray) -> np.ndarray:
        """Weighted RMSE: sqrt(Σ w×(|pred-t|/|t|)² / Σw) — mirrors lammps_interface."""
        if physics_features:
            X_feat = feat_builder.transform(X)
        else:
            X_feat = X
        sq_sum     = np.zeros(X.shape[0])
        weight_sum = 0.0
        for prop, info in active_targets.items():
            if prop not in ensembles:
                continue
            target = float(info["value"])
            weight = float(info.get("weight", 1.0))
            pred   = ensembles[prop].predict_mean(X_feat)   # (N,)
            rel_err    = np.abs(pred - target) / (abs(target) + 1e-10)
            sq_sum    += weight * rel_err ** 2
            weight_sum += weight
        return np.sqrt(sq_sum / max(weight_sum, 1e-12))

    return obj


# ============================================================================
# Sobol total-order sensitivity
# ============================================================================

def compute_sobol_sensitivity(obj_fn, param_lo, param_hi,
                              n_samples: int, seed: int) -> np.ndarray:
    """
    Estimate Sobol total-order sensitivity indices via Saltelli's estimator.

    Saltelli (2002): T_i = E[Var(Y | X_~i)] / Var(Y)
                         ≈ 1/(2N) Σ (f_B - f_A_Bi)²  /  Var(Y)

    Parameters
    ----------
    n_samples : base sample size N (total function evaluations = N × (D+2))
    Returns   : T_i array of shape (D,)
    """
    D   = len(param_lo)
    rng = np.random.default_rng(seed)

    # Saltelli sample matrices A and B
    sampler = qmc.Sobol(d=2 * D, scramble=True, seed=seed)
    n_pow2  = int(2 ** np.ceil(np.log2(n_samples)))
    raw     = sampler.random(n_pow2)
    A_unit  = raw[:, :D]
    B_unit  = raw[:, D:]
    A = qmc.scale(A_unit, param_lo, param_hi)
    B = qmc.scale(B_unit, param_lo, param_hi)

    f_A = obj_fn(A)
    f_B = obj_fn(B)
    var_Y = np.var(np.concatenate([f_A, f_B]))
    if var_Y < 1e-15:
        return np.zeros(D)

    T = np.zeros(D)
    for i in range(D):
        # A_Bi: matrix A with column i replaced by column i from B
        A_Bi       = A.copy()
        A_Bi[:, i] = B[:, i]
        f_ABi      = obj_fn(A_Bi)
        # Saltelli total-order estimator
        T[i] = np.mean((f_B - f_ABi) ** 2) / (2.0 * var_Y)

    # Clip to [0, 1]; Sobol indices can be slightly negative due to sampling
    return np.clip(T, 0.0, 1.0)


# ============================================================================
# Gradient-based sensitivity
# ============================================================================

def compute_grad_sensitivity(obj_fn, param_lo, param_hi,
                             n_samples: int, seed: int) -> np.ndarray:
    """
    Mean absolute finite-difference gradient over a Sobol sample pool.

    S_i = mean_x |∂f/∂x_i| × (hi_i - lo_i)
    The (hi-lo) factor normalises across parameters with different scales.
    """
    D = len(param_lo)
    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    n_pow2  = int(2 ** np.ceil(np.log2(n_samples)))
    X = qmc.scale(sampler.random(n_pow2), param_lo, param_hi)

    span = param_hi - param_lo
    eps  = np.maximum(1e-4 * span, 1e-8)

    grads = np.zeros((n_pow2, D))
    for i in range(D):
        Xp = X.copy(); Xp[:, i] += eps[i]
        Xm = X.copy(); Xm[:, i] -= eps[i]
        grads[:, i] = np.abs((obj_fn(Xp) - obj_fn(Xm)) / (2.0 * eps[i])) * span[i]

    return grads.mean(axis=0)


# ============================================================================
# Plotting
# ============================================================================

def plot_sensitivity(param_names: List[str],
                     sobol:  Optional[np.ndarray],
                     grads:  Optional[np.ndarray],
                     out_path: str):
    """Horizontal bar chart ranked by Sobol index (or gradient if no Sobol)."""
    n = len(param_names)
    if sobol is not None:
        scores = sobol
        xlabel = "Sobol total-order index"
    else:
        scores = grads
        xlabel = "Mean |∂obj/∂param| × range"

    rank = np.argsort(scores)[::-1]
    ranked_names  = [param_names[i] for i in rank]
    ranked_scores = scores[rank]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * n)))
    bars = ax.barh(range(n), ranked_scores[::-1], color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ranked_names[::-1], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title("Parameter Sensitivity Ranking")
    ax.axvline(0, color="#888", lw=0.7)

    # Annotate values
    for bar, val in zip(bars, ranked_scores[::-1]):
        ax.text(val + max(ranked_scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # If both methods available, add gradient as error-bar overlay
    if sobol is not None and grads is not None:
        grad_ranked = grads[rank][::-1]
        grad_norm   = grad_ranked / (grad_ranked.max() + 1e-10) * ranked_scores.max()
        ax.plot(grad_norm, range(n), "D", color="#DD8452",
                ms=5, alpha=0.7, label="Grad (normalised)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parameter Sensitivity Analysis via trained NN ensemble")
    parser.add_argument("--config",    default="configs/recipes/cluster_full.yaml")
    parser.add_argument("--nn-dir",    default=None,
                        help="NNSurrogate output dir with forward_nn.pt. "
                             "Default: auto-discover newest nn_output_*/")
    parser.add_argument("--method",    choices=["sobol", "grad", "both"],
                        default="both")
    parser.add_argument("--n-samples", type=int, default=2048,
                        help="Base sample size (Sobol: D+2 evals total; "
                             "grad: n_samples evals). Default: 2048")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    # ── Load config ─────────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── Resolve nn_dir ───────────────────────────────────────────────────────
    nn_dir = args.nn_dir
    if nn_dir is None:
        import glob as _g
        found = sorted(_g.glob("nn_output_*/forward_nn.pt"),
                       key=os.path.getmtime, reverse=True)
        if found:
            nn_dir = os.path.dirname(found[0])
            print(f"Auto-selected NN dir: {nn_dir}")
        else:
            print("ERROR: --nn-dir not given and no nn_output_*/forward_nn.pt found.")
            sys.exit(1)

    pt_path = os.path.join(nn_dir, "forward_nn.pt")
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found."); sys.exit(1)

    # ── Load ensemble ────────────────────────────────────────────────────────
    print(f"Loading ensemble from {pt_path}")
    ensembles, feat_builder, param_names, feat_names, meta = \
        load_ensemble_from_file(pt_path)

    targets_cfg     = meta.get("targets", config.get("targets", {}))
    compute_surface = meta.get("compute_surface",
                               config["lammps"]["compute_surface"])
    physics_features = meta["physics_features"]

    print(f"  Properties : {list(ensembles.keys())}")
    print(f"  Parameters : {param_names}")

    # ── Build bounds ─────────────────────────────────────────────────────────
    param_space = ForceFieldOptimizer._build_param_space(config)
    param_lo    = np.array([p[1] for p in param_space])
    param_hi    = np.array([p[2] for p in param_space])

    # Align param_names from model with param_space
    ps_names = [p[0] for p in param_space]
    if ps_names != param_names:
        print("  WARNING: param_names in model differ from current config; "
              "using model's param_names and bounds from model.")
        param_lo = np.array(meta["param_lo"])
        param_hi = np.array(meta["param_hi"])
        param_names = meta["param_names"]

    # ── Build objective function ─────────────────────────────────────────────
    obj_fn = _build_obj_fn(ensembles, feat_builder, targets_cfg,
                           physics_features, compute_surface)

    n = args.n_samples
    print(f"\nSensitivity analysis  method={args.method}  n_samples={n}")

    sobol_S: Optional[np.ndarray] = None
    grad_S:  Optional[np.ndarray] = None

    if args.method in ("sobol", "both"):
        print(f"  Running Sobol ({n*(len(param_names)+2)} total evals)...")
        sobol_S = compute_sobol_sensitivity(obj_fn, param_lo, param_hi, n, args.seed)
        df_sobol = pd.DataFrame({"parameter": param_names, "sobol_total": sobol_S})
        df_sobol = df_sobol.sort_values("sobol_total", ascending=False)
        sobol_path = os.path.join(nn_dir, "sensitivity_sobol.csv")
        df_sobol.to_csv(sobol_path, index=False)
        print(f"\n  Sobol total-order indices (top-5):")
        print(df_sobol.head(5).to_string(index=False))
        print(f"  Saved: {sobol_path}")

    if args.method in ("grad", "both"):
        print(f"\n  Running gradient sensitivity ({n} evals per param × {len(param_names)} params)...")
        grad_S = compute_grad_sensitivity(obj_fn, param_lo, param_hi, n, args.seed)
        df_grad = pd.DataFrame({"parameter": param_names, "grad_sensitivity": grad_S})
        df_grad = df_grad.sort_values("grad_sensitivity", ascending=False)
        grad_path = os.path.join(nn_dir, "sensitivity_grad.csv")
        df_grad.to_csv(grad_path, index=False)
        print(f"\n  Gradient sensitivity (top-5):")
        print(df_grad.head(5).to_string(index=False))
        print(f"  Saved: {grad_path}")

    # ── Plot ───────────────────────────────────────────────
