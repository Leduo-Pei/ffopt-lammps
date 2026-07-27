"""
viz/al_plots.py -- Active learning convergence figure (v8)

Functions
---------
  plot_active_learning_curve(al_json, plot_cfg, figs_dir)
      Multi-panel convergence plot across AL rounds:
        Panel 1 (left)  : max + mean ensemble std (uncertainty) vs. round
        Panel 2 (right) : best LAMMPS objective + test R² vs. round

Saves:
  {figs_dir}/active_learning_curve.pdf
  {figs_dir}/active_learning_curve.png
  {figs_dir}/active_learning_curve_data.csv
"""

import json
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import get_color, get_figsize, save_fig


def plot_active_learning_curve(al_json:  Union[str, Path],
                               plot_cfg: dict,
                               figs_dir: Union[str, Path]):
    """
    Plot active learning convergence metrics across rounds.

    Reads active_learning_output/active_learning_history.json written
    by active_learning.py._save_history().

    Panels
    ------
    Left  : ensemble uncertainty (max_std and mean_std before/after retrain)
            Horizontal dashed line = uncertainty_threshold
    Right : best LAMMPS objective (cumulative minimum) + test R²
    """
    with open(al_json) as fh:
        hist = json.load(fh)

    rounds_data = hist.get("rounds", [])
    if not rounds_data:
        print("    plot_active_learning_curve: history is empty; skipping")
        return

    # Extract per-round metrics (use .get with nan fallback for robustness)
    rounds        = [r["round"]                             for r in rounds_data]
    max_std_b     = [r.get("max_std_before",   np.nan)     for r in rounds_data]
    mean_std_b    = [r.get("mean_std_before",  np.nan)     for r in rounds_data]
    max_std_a     = [r.get("max_std_after",    np.nan)     for r in rounds_data]
    mean_std_a    = [r.get("mean_std_after",   np.nan)     for r in rounds_data]
    best_obj      = [r.get("best_lammps_obj_so_far", np.nan) for r in rounds_data]
    test_r2       = [r.get("test_r2",          np.nan)     for r in rounds_data]
    n_new_evals   = [r.get("n_new_evals",      0)          for r in rounds_data]
    early_stopped = hist.get("early_stopped",  False)
    final_round   = hist.get("final_round",    rounds[-1] if rounds else 0)

    # Cumulative evaluation count for x-axis annotation
    cum_evals = np.cumsum(n_new_evals).tolist()

    # ── Figure: 2 side-by-side panels ────────────────────────────────────────
    figsize = get_figsize(plot_cfg, "al_curve", [10, 4.5])
    fig, (ax_std, ax_obj) = plt.subplots(1, 2, figsize=figsize)

    rounds_arr = np.array(rounds, dtype=float)

    # ── Panel 1: Uncertainty ─────────────────────────────────────────────────
    c_max  = get_color(plot_cfg, "al_max_std",  "#DD8452")
    c_mean = get_color(plot_cfg, "al_mean_std", "#4C72B0")

    # Before-retrain: solid; after-retrain: dashed
    ax_std.plot(rounds_arr, max_std_b, "o-",
                color=c_max, lw=1.5, ms=5, label="max std (before retrain)")
    ax_std.plot(rounds_arr, mean_std_b, "s-",
                color=c_mean, lw=1.5, ms=5, label="mean std (before retrain)")

    if any(not np.isnan(v) for v in max_std_a):
        ax_std.plot(rounds_arr, max_std_a, "o--",
                    color=c_max, lw=1.0, ms=4, alpha=0.6,
                    label="max std (after retrain)")
        ax_std.plot(rounds_arr, mean_std_a, "s--",
                    color=c_mean, lw=1.0, ms=4, alpha=0.6,
                    label="mean std (after retrain)")

    # Uncertainty threshold horizontal line
    # Try to read from history metadata, fall back to a reasonable default
    threshold = hist.get("uncertainty_threshold", None)
    if threshold is None and max_std_b:
        finite_stds = [v for v in max_std_b if np.isfinite(v)]
        threshold   = min(finite_stds) * 0.5 if finite_stds else None
    if threshold is not None:
        ax_std.axhline(threshold, ls=":", lw=1.2,
                       color=get_color(plot_cfg, "refline", "#888888"),
                       label=f"threshold = {threshold:.3f}")

    ax_std.set_xlabel("AL round")
    ax_std.set_ylabel("Ensemble std (epistemic uncertainty)")
    ax_std.set_title("Uncertainty Convergence")
    ax_std.set_xticks(rounds)
    ax_std.legend(fontsize=8, loc="upper right")

    # ── Panel 2: Best objective + R² ─────────────────────────────────────────
    ax2 = ax_obj.twinx()

    c_obj = get_color(plot_cfg, "al_best_obj", "#C44E52")
    c_r2  = get_color(plot_cfg, "al_r2",       "#55A868")

    # Running minimum of best_lammps_obj_so_far (already cumulative from AL)
    ax_obj.plot(rounds_arr, best_obj, "D-",
                color=c_obj, lw=1.8, ms=6, label="Best LAMMPS obj (cumul.)")

    finite_r2 = [v for v in test_r2 if np.isfinite(v)]
    if finite_r2:
        ax2.plot(rounds_arr, test_r2, "^--",
                 color=c_r2, lw=1.4, ms=5, alpha=0.85,
                 label="Test R²")
        ax2.set_ylabel("Test R²", color=c_r2)
        ax2.tick_params(axis="y", labelcolor=c_r2)
        ax2.set_ylim(max(0, min(finite_r2) - 0.05), 1.02)

    ax_obj.set_xlabel("AL round")
    ax_obj.set_ylabel("Objective (weighted RMSE)", color=c_obj)
    ax_obj.tick_params(axis="y", labelcolor=c_obj)
    ax_obj.set_title("Objective + Surrogate Quality")
    ax_obj.set_xticks(rounds)

    # Combine legends from both axes
    lines1, labs1 = ax_obj.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax_obj.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower left")

    # Overall title
    status = "early stopped" if early_stopped else f"all {final_round} rounds"
    fig.suptitle(
        f"Active Learning Convergence  ({status}  |  "
        f"total AL evals = {hist.get('total_al_evals', '?')})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Data CSV ─────────────────────────────────────────────────────────────
    data_df = pd.DataFrame({
        "round":           rounds,
        "n_new_evals":     n_new_evals,
        "cum_evals":       cum_evals,
        "max_std_before":  max_std_b,
        "mean_std_before": mean_std_b,
        "max_std_after":   max_std_a,
        "mean_std_after":  mean_std_a,
        "best_lammps_obj": best_obj,
        "test_r2":         test_r2,
    })
    save_fig(fig, "active_learning_curve", figs_dir, plot_cfg, data_df)
