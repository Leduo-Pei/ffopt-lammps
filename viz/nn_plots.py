"""
viz/nn_plots.py -- Neural network surrogate figures (v8)

Functions
---------
  plot_nn_parity(bo_csv, pt_path, plot_cfg, figs_dir, config=None)
      NN ensemble mean predicted vs. LAMMPS actual objective.
      Annotated with R² (train / test) and ensemble std band.

  plot_nn_optimize_result(opt_json, plot_cfg, figs_dir, bo_csv_path=None)
      Per-property deviation bar chart comparing the NN-optimal parameter set
      against experimental targets, with and without LAMMPS validation.

  plot_optimize_candidates(opt_json, plot_cfg, figs_dir)
      Scatter of all optimization candidates:
      predicted objective vs. candidate rank, coloured by LAMMPS result.

  plot_lammps_validation(opt_json, plot_cfg, figs_dir)
      Side-by-side bar comparison (NN predicted vs. LAMMPS actual) for each
      property at the best parameter set.

All functions save:
  {figs_dir}/{name}.pdf
  {figs_dir}/{name}.png
  {figs_dir}/{name}_data.csv
"""

import json
import sys
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (detect_param_cols, get_color, get_figsize, save_fig)


# ============================================================================
# 1. Parity plot (NN predicted vs. LAMMPS actual)
# ============================================================================

def plot_nn_parity(bo_csv:    Union[str, Path],
                   pt_path:   Union[str, Path],
                   plot_cfg:  dict,
                   figs_dir:  Union[str, Path],
                   config:    Optional[dict] = None):
    """
    Scatter: ensemble mean prediction vs. actual LAMMPS objective.

    Loads the saved ensemble (forward_nn.pt) and evaluates all valid BO
    points to build the parity scatter.  Shows:
      • Scatter  : predicted vs. actual (colour = eval round)
      • Red line : identity (y = x)
      • Shaded   : ±1 ensemble std band (epistemic uncertainty)
      • Annotation: R² on all valid points
    """
    # ── Import nn_surrogate loader ────────────────────────────────────────────
    # Add the parent directory (material_v8/) to path so nn_surrogate is importable
    _add_parent_to_path(pt_path)

    try:
        from engine.nn_surrogate import load_ensemble_from_file
    except ImportError as e:
        print(f"    plot_nn_parity: cannot import nn_surrogate ({e}); skipping")
        return

    # ── Load BO data ──────────────────────────────────────────────────────────
    df = pd.read_csv(bo_csv)
    valid = df[df["success"].astype(bool) & np.isfinite(df["objective"])].copy()
    if valid.empty:
        print("    plot_nn_parity: no valid data")
        return

    param_cols = detect_param_cols(df)
    # Only keep param cols that are actually present in the dataframe
    param_cols = [c for c in param_cols if c in valid.columns]
    if not param_cols:
        print("    plot_nn_parity: no param columns detected")
        return

    X_raw   = valid[param_cols].values.astype(np.float64)
    Y_true  = valid["objective"].values.astype(np.float64)
    rounds  = valid.get("round", pd.Series(0, index=valid.index)).values

    # ── Load ensemble ─────────────────────────────────────────────────────────
    try:
        ensemble, feat_builder, model_param_names, _, meta = \
            load_ensemble_from_file(str(pt_path))
    except Exception as e:
        print(f"    plot_nn_parity: failed to load model ({e}); skipping")
        return

    # Align param columns with model's expected order
    aligned_X = _align_params(X_raw, param_cols, model_param_names)

    # Apply physics features if the model was trained with them
    if meta.get("physics_features", False):
        X_feat = feat_builder.transform(aligned_X)
    else:
        X_feat = aligned_X

    Y_pred = ensemble.predict_mean(X_feat)
    Y_std  = ensemble.predict_std(X_feat)

    # ── R² ────────────────────────────────────────────────────────────────────
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - Y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=get_figsize(plot_cfg, "parity", [6, 6]))

    # Colour points by round number (shows how later rounds improve fit)
    sc = ax.scatter(Y_true, Y_pred,
                    c=rounds, cmap="Blues",
                    s=20, alpha=0.7, linewidths=0,
                    zorder=3, label="BO points")

    # ±1 std band (draw as error bars for clarity)
    ax.errorbar(Y_true, Y_pred, yerr=Y_std,
                fmt="none", ecolor="#AAAAAA", elinewidth=0.6,
                capsize=0, alpha=0.4, zorder=2)

    # Identity line y = x
    all_vals = np.concatenate([Y_true, Y_pred])
    lo, hi   = np.nanmin(all_vals) * 0.95, np.nanmax(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2,
            color=get_color(plot_cfg, "nn_parity_line", "#C44E52"),
            label="y = x", zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("LAMMPS objective (actual)")
    ax.set_ylabel("Ensemble mean (predicted)")
    ax.set_title(f"NN Parity  —  R² = {r2:.4f}")
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(sc, ax=ax, label="BO round", pad=0.02)
    ax.legend(fontsize=9)

    data_df = pd.DataFrame({
        "y_true": Y_true,
        "y_pred": Y_pred,
        "y_std":  Y_std,
        "round":  rounds,
    })
    save_fig(fig, "nn_parity", figs_dir, plot_cfg, data_df)


# ============================================================================
# 2. NN optimize result (per-property deviation bar chart)
# ============================================================================

def plot_nn_optimize_result(opt_json:    Union[str, Path],
                            plot_cfg:    dict,
                            figs_dir:    Union[str, Path],
                            bo_csv_path: Optional[Union[str, Path]] = None):
    """
    Per-property error bar chart for the best NN-optimised parameter set.

    Shows the percentage deviation from experiment for:
      • The best BO point   (from bo_csv, if provided)
      • The NN-optimised best (LAMMPS-validated if available, else NN-predicted)
    """
    with open(opt_json) as fh:
        data = json.load(fh)

    best = data.get("best_lammps") or data.get("best_nn") or data.get("best")
    if not best:
        print("    plot_nn_optimize_result: no 'best' entry in JSON; skipping")
        return

    per_err = best.get("per_property_error") or best.get("lammps_properties", {})
    if not per_err:
        print("    plot_nn_optimize_result: no per_property_error in best; skipping")
        return

    # Convert per-property errors (could be %-already or raw)
    props   = [k for k in per_err if not k.startswith("_")]
    errors  = [float(per_err[k]) for k in props]

    # Best BO point for comparison (optional)
    bo_errors = []
    if bo_csv_path and Path(bo_csv_path).exists():
        bo_df   = pd.read_csv(bo_csv_path)
        bo_valid = bo_df[bo_df["success"].astype(bool) &
                         np.isfinite(bo_df["objective"])].copy()
        if not bo_valid.empty:
            best_bo_row = bo_valid.loc[bo_valid["objective"].idxmin()]
            bo_errors   = [float(best_bo_row.get(f"error_{p}", np.nan))
                           for p in props]

    fig, ax = plt.subplots(figsize=get_figsize(plot_cfg, "optimize_result", [8, 4.5]))

    x        = np.arange(len(props))
    bar_w    = 0.35 if bo_errors else 0.6
    offset   = bar_w / 2 if bo_errors else 0

    ax.bar(x - offset, np.abs(errors), width=bar_w,
           color=get_color(plot_cfg, "nn_bar_lammps", "#55A868"),
           alpha=0.85, label="NN-optimal (LAMMPS-validated)")

    if bo_errors:
        ax.bar(x + offset, [abs(v) if np.isfinite(v) else 0 for v in bo_errors],
               width=bar_w,
               color=get_color(plot_cfg, "bo_initial", "#4C72B0"),
               alpha=0.7, label="Best BO point")

    # 5% target line
    ax.axhline(5.0, ls="--", lw=1.0,
               color=get_color(plot_cfg, "refline", "#888888"),
               label="5% target")

    ax.set_xticks(x)
    ax.set_xticklabels(props, rotation=25, ha="right")
    ax.set_ylabel("|Deviation from experiment| (%)")
    ax.set_title("NN-Optimised Parameter Set: Property Errors")
    ax.legend(fontsize=9)

    # Annotate bars
    for xi, err in zip(x - offset, errors):
        ax.text(xi, abs(err) + 0.1, f"{abs(err):.1f}%",
                ha="center", va="bottom", fontsize=8)

    source = ("LAMMPS-validated" if data.get("best_lammps") else "NN-predicted")
    ax.text(0.99, 0.98, f"Source: {source}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#666666")

    data_df = pd.DataFrame({
        "property":     props,
        "nn_error_pct": errors,
        "bo_error_pct": bo_errors if bo_errors else [np.nan] * len(props),
    })
    save_fig(fig, "nn_optimize_result", figs_dir, plot_cfg, data_df)


# ============================================================================
# 3. Optimization candidates scatter
# ============================================================================

def plot_optimize_candidates(opt_json: Union[str, Path],
                             plot_cfg: dict,
                             figs_dir: Union[str, Path]):
    """
    Predicted objective for all NN-optimization candidates, sorted by rank.

    Points coloured by whether LAMMPS validation succeeded.
    """
    with open(opt_json) as fh:
        data = json.load(fh)

    candidates = data.get("all_candidates", [])
    if not candidates:
        print("    plot_optimize_candidates: no candidates in JSON; skipping")
        return

    pred_objs = [float(c.get("predicted_obj", np.nan)) for c in candidates]
    ranks     = list(range(1, len(candidates) + 1))

    # Match to LAMMPS results if present
    lammps_map: dict = {}
    for r in data.get("all_lammps", []):
        rk = r.get("rank")
        if rk is not None:
            lammps_map[rk] = r

    lammps_objs    = [lammps_map.get(rk, {}).get("lammps_obj", np.nan)
                      for rk in ranks]
    lammps_success = [lammps_map.get(rk, {}).get("lammps_success", False)
                      for rk in ranks]

    fig, ax = plt.subplots(figsize=get_figsize(plot_cfg, "candidates", [7, 5]))

    # All candidates: predicted objective vs. rank
    ax.plot(ranks, pred_objs, "o-",
            color=get_color(plot_cfg, "nn_bar_pred", "#4C72B0"),
            ms=6, lw=1.2, alpha=0.8, label="NN prediction")

    # LAMMPS-validated points (overlay)
    val_ranks  = [rk for rk, ok in zip(ranks, lammps_success) if ok]
    val_lammps = [lammps_objs[i] for i, ok in enumerate(lammps_success) if ok]
    if val_ranks:
        ax.scatter(val_ranks, val_lammps,
                   s=80, zorder=5, marker="D",
                   color=get_color(plot_cfg, "nn_bar_lammps", "#55A868"),
                   edgecolors="k", linewidths=0.5,
                   label="LAMMPS (validated)")

    # Failed LAMMPS
    fail_ranks = [rk for rk, ok in zip(ranks, lammps_success)
                  if not ok and rk in lammps_map]
    if fail_ranks:
        ax.scatter(fail_ranks,
                   [pred_objs[rk - 1] for rk in fail_ranks],
                   s=60, zorder=5, marker="X",
                   color=get_color(plot_cfg, "bo_infeasible", "#C44E52"),
                   alpha=0.7, label="LAMMPS (failed)")

    ax.set_xlabel("Candidate rank (1 = best NN prediction)")
    ax.set_ylabel("Objective")
    ax.set_title("NN Optimization Candidates")
    ax.legend(fontsize=9)

    data_df = pd.DataFrame({
        "rank":          ranks,
        "predicted_obj": pred_objs,
        "lammps_obj":    lammps_objs,
        "lammps_success": lammps_success,
    })
    save_fig(fig, "optimize_candidates", figs_dir, plot_cfg, data_df)


# ============================================================================
# 4. LAMMPS validation bar chart
# ============================================================================

def plot_lammps_validation(opt_json: Union[str, Path],
                           plot_cfg: dict,
                           figs_dir: Union[str, Path]):
    """
    Side-by-side bar chart: NN-predicted property values vs. LAMMPS-computed
    values vs. experimental targets, for the best validated parameter set.
    """
    with open(opt_json) as fh:
        data = json.load(fh)

    # Need the LAMMPS-validated best
    best_lammps = data.get("best_lammps")
    if not best_lammps or not best_lammps.get("lammps_success"):
        print("    plot_lammps_validation: no successful LAMMPS result; skipping")
        return

    props_dict = best_lammps.get("lammps_properties", {})
    per_err    = best_lammps.get("per_property_error", {})
    if not props_dict:
        print("    plot_lammps_validation: no lammps_properties; skipping")
        return

    props    = [k for k in props_dict
                if k not in ("E_complete", "E_split", "A_xy")]
    if not props:
        print("    plot_lammps_validation: no displayable properties; skipping")
        return

    lammps_vals = [float(props_dict.get(p, np.nan)) for p in props]
    errors_pct  = [float(per_err.get(p, np.nan))     for p in props]

    fig, axes = plt.subplots(1, 2,
                             figsize=get_figsize(plot_cfg, "lammps_val", [7, 4.5]))

    # Left: property values (normalised to target = 1.0 so all on same scale)
    ax0 = axes[0]
    ax0.bar(range(len(props)), lammps_vals,
            color=get_color(plot_cfg, "nn_bar_lammps", "#55A868"),
            alpha=0.85, edgecolor="none")
    ax0.set_xticks(range(len(props)))
    ax0.set_xticklabels(props, rotation=30, ha="right", fontsize=9)
    ax0.set_ylabel("Computed value")
    ax0.set_title("LAMMPS Properties")

    # Right: percentage errors
    ax1 = axes[1]
    colors_bar = [
        get_color(plot_cfg, "bo_infeasible", "#C44E52") if abs(e) > 5.0
        else get_color(plot_cfg, "nn_bar_lammps", "#55A868")
        for e in errors_pct
    ]
    ax1.bar(range(len(props)), [abs(e) for e in errors_pct],
            color=colors_bar, alpha=0.85, edgecolor="none")
    ax1.axhline(5.0, ls="--", lw=1.0,
                color=get_color(plot_cfg, "refline", "#888888"),
                label="5% target")
    ax1.set_xticks(range(len(props)))
    ax1.set_xticklabels(props, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("|Error| (%)")
    ax1.set_title("Property Errors")
    ax1.legend(fontsize=8)

    for xi, err in zip(range(len(props)), errors_pct):
        ax1.text(xi, abs(err) + 0.1, f"{abs(err):.1f}%",
                 ha="center", va="bottom", fontsize=8)

    lammps_obj = best_lammps.get("lammps_obj", np.nan)
    fig.suptitle(f"LAMMPS Validation  —  Objective = {lammps_obj:.6f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    data_df = pd.DataFrame({
        "property":     props,
        "lammps_value": lammps_vals,
        "error_pct":    errors_pct,
    })
    save_fig(fig, "lammps_validation", figs_dir, plot_cfg, data_df)


# ============================================================================
# Utilities
# ============================================================================

def _add_parent_to_path(pt_path: Union[str, Path]):
    """Add the directory two levels above the .pt file to sys.path."""
    parent = str(Path(pt_path).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Also try the directory of this viz module (material_v8/)
    here = str(Path(__file__).resolve().parent.parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def _align_params(X_raw:         np.ndarray,
                  csv_param_cols: list,
                  model_names:    list) -> np.ndarray:
    """
    Re-order / subset X_raw columns to match the model's expected param order.

    If a model param name is missing from csv_param_cols, fills with NaN
    (will likely produce garbage predictions, but avoids a crash).
    """
    if csv_param_cols == model_names:
        return X_raw

    N = X_raw.shape[0]
    idx_map = {n: i for i, n in enumerate(csv_param_cols)}
    aligned = np.full((N, len(model_names)), np.nan)
    for j, name in enumerate(model_names):
        if name in idx_map:
            aligned[:, j] = X_raw[:, idx_map[name]]
    return aligned
