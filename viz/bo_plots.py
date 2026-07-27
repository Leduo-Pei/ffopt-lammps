"""
viz/bo_plots.py -- Bayesian Optimisation result figures (v8)

Functions
---------
  plot_bo_convergence(bo_csv, plot_cfg, figs_dir)
      Convergence curve: best objective vs. cumulative evaluation count.
      Shows initial LHS phase vs. BO acquisition phase.

  plot_bo_param_space(bo_csv, plot_cfg, figs_dir)
      Parameter space scatter plots.
      N ≤ 6 params : full pair-plot grid (all pairs).
      N 7–12 params: pair-plot of top-6 params most correlated with objective.
      N ≥ 13 params: 1-D marginal distributions + correlation bar chart.
      Fully dynamic — reads param names from CSV columns, no hardcoding.

  plot_feasibility_map(bo_csv, plot_cfg, figs_dir)
      2-D scatter of the two params with strongest feasibility signal,
      coloured green (feasible) / red (infeasible).
      Falls back to a univariate feasibility rate plot when N=1.

All functions save:
  {figs_dir}/{name}.pdf
  {figs_dir}/{name}.png
  {figs_dir}/{name}_data.csv  (raw data used for the figure)
"""

from pathlib import Path
from typing import Union
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from .style import (detect_param_cols, get_color, get_figsize,
                    save_fig)

# Pair-plot max params before switching to reduced / marginal-only view
_FULL_PAIR_MAX    = 6    # ≤6 → full pair grid
_REDUCED_PAIR_MAX = 12   # 7-12 → top-6 pairs; ≥13 → marginals only


# ============================================================================
# 1. Convergence curve
# ============================================================================

def plot_bo_convergence(bo_csv:   Union[str, Path],
                        plot_cfg: dict,
                        figs_dir: Union[str, Path]):
    """
    Best objective vs. cumulative evaluation number.

    Shows:
      • Grey dots  : all evaluated points (valid + infeasible)
      • Blue line  : running best (valid only)
      • Vertical dashed line: boundary between initial LHS and BO acquisition
    """
    df = pd.read_csv(bo_csv)

    # Running best (valid = success + finite objective)
    valid_mask = df["success"].astype(bool) & np.isfinite(df["objective"])
    objs_valid = df["objective"].copy()
    objs_valid[~valid_mask] = np.inf

    running_best = np.minimum.accumulate(objs_valid.values)
    evals = np.arange(1, len(df) + 1)

    # Detect LHS / BO boundary: round == 0 entries are labelled "initial"
    n_initial = int((df["label"] == "initial").sum()) if "label" in df.columns else 0

    fig, ax = plt.subplots(figsize=get_figsize(plot_cfg, "convergence", [8, 4]))

    # All objective dots (clipped to finite range for display)
    valid_objs = df.loc[valid_mask, "objective"].values
    y_max_disp = (np.percentile(valid_objs, 98) * 1.5
                  if valid_objs.size > 0 else 10.0)
    display_obj = np.clip(df["objective"].values, None, y_max_disp)
    ax.scatter(evals[valid_mask],  display_obj[valid_mask],
               s=14, alpha=0.4,
               color=get_color(plot_cfg, "bo_valid",      "#55A868"),
               label="Valid")
    ax.scatter(evals[~valid_mask], display_obj[~valid_mask],
               s=14, alpha=0.25, marker="x",
               color=get_color(plot_cfg, "bo_infeasible", "#C44E52"),
               label="Infeasible")

    # Running best line
    rb_finite = np.where(np.isfinite(running_best), running_best, np.nan)
    ax.plot(evals, rb_finite, lw=1.8,
            color=get_color(plot_cfg, "bo_best", "#DD8452"),
            label="Running best")

    # Mark the true best point
    if np.any(np.isfinite(rb_finite)):
        best_idx = np.nanargmin(rb_finite)
        ax.scatter(evals[best_idx], rb_finite[best_idx],
                   s=120, zorder=5, marker="*",
                   color=get_color(plot_cfg, "bo_best", "#DD8452"),
                   edgecolors="k", linewidths=0.5,
                   label=f"Best = {rb_finite[best_idx]:.4f}")

    # Vertical separator between initial LHS and BO phase
    if n_initial > 0:
        ax.axvline(n_initial + 0.5, ls="--", lw=1.0,
                   color=get_color(plot_cfg, "refline", "#888888"),
                   label=f"LHS end (n={n_initial})")
        ax.text(n_initial + 1.0, ax.get_ylim()[1] * 0.97, "← BO acquisition",
                fontsize=9, va="top", alpha=0.7)

    ax.set_xlabel("Cumulative evaluations")
    ax.set_ylabel("Objective (weighted RMSE)")
    ax.set_title("BO Convergence")
    ax.legend(loc="upper right", framealpha=0.85)

    # Data CSV: eval index, objective, running_best, success, round
    data_df = pd.DataFrame({
        "eval":         evals,
        "objective":    df["objective"].values,
        "running_best": rb_finite,
        "success":      df["success"].astype(bool).values,
        "round":        df.get("round", pd.Series(0, index=df.index)).values,
    })
    save_fig(fig, "bo_convergence", figs_dir, plot_cfg, data_df)


# ============================================================================
# 2. Parameter space
# ============================================================================

def plot_bo_param_space(bo_csv:   Union[str, Path],
                        plot_cfg: dict,
                        figs_dir: Union[str, Path]):
    """
    Scatter plots of the BO parameter space coloured by objective.

    Adapts to the number of free parameters:
      N ≤ 6  : full pair-plot grid (N×N panels)
      7-12   : pair-plot of top-6 params most correlated with objective
      ≥13    : 1-D marginal distributions + correlation ranking bar chart
    """
    df    = pd.read_csv(bo_csv)
    valid = df[df["success"].astype(bool) & np.isfinite(df["objective"])].copy()

    param_cols = detect_param_cols(df)
    n_params   = len(param_cols)

    if valid.empty or n_params == 0:
        print("    plot_bo_param_space: no valid data or no param columns found")
        return

    if n_params <= _FULL_PAIR_MAX:
        _pair_plot(valid, param_cols, plot_cfg, figs_dir, tag="full")

    elif n_params <= _REDUCED_PAIR_MAX:
        # Select top-6 params by |Pearson correlation| with objective
        corr_vals  = valid[param_cols].corrwith(valid["objective"]).abs()
        top_params = corr_vals.nlargest(6).index.tolist()
        _pair_plot(valid, top_params, plot_cfg, figs_dir, tag="top6",
                   subtitle=f"Top-6 of {n_params} params by |corr| with objective")

    else:
        # N ≥ 13: marginal histograms + correlation bar chart
        _marginals_and_corr(valid, param_cols, plot_cfg, figs_dir)


def _pair_plot(df:       pd.DataFrame,
               params:   list,
               plot_cfg: dict,
               figs_dir: Union[str, Path],
               tag:      str = "",
               subtitle: str = ""):
    """
    Lower-triangle pair-plot: scatter coloured by objective + diagonal histogram.
    """
    n = len(params)
    figsize = get_figsize(plot_cfg, "param_space", [9, 9])
    # Scale figure size linearly with N panels
    scale  = max(1.0, n / 4.0)
    figw   = min(figsize[0] * scale, 20.0)
    figh   = min(figsize[1] * scale, 20.0)

    fig, axes = plt.subplots(n, n, figsize=(figw, figh))
    if n == 1:
        axes = np.array([[axes]])

    objs  = df["objective"].values
    vmin, vmax = np.nanpercentile(objs, [2, 98])
    cmap  = plt.cm.viridis_r

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if col == row:
                # Diagonal: histogram of this param (valid vs. infeasible)
                ax.hist(df[params[row]].values, bins=20, density=True,
                        color=get_color(plot_cfg, "bo_valid", "#55A868"),
                        alpha=0.7, edgecolor="none")
                ax.set_yticks([])
            else:
                sc = ax.scatter(
                    df[params[col]].values, df[params[row]].values,
                    c=objs, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=12, alpha=0.65, linewidths=0,
                )
            # Labels only on outer edges
            if row == n - 1:
                ax.set_xlabel(params[col], fontsize=8, labelpad=2)
            if col == 0 and row != 0:
                ax.set_ylabel(params[row], fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.1, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm      = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Objective")

    title = "BO Parameter Space"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, y=1.01, fontsize=13)

    stem = f"bo_param_space_{tag}" if tag else "bo_param_space"
    data_df = df[params + ["objective", "success"]].copy()
    save_fig(fig, stem, figs_dir, plot_cfg, data_df)


def _marginals_and_corr(df:       pd.DataFrame,
                        params:   list,
                        plot_cfg: dict,
                        figs_dir: Union[str, Path]):
    """
    For high-dimensional spaces (N ≥ 13):
    Left panels: 1-D marginal histograms for each param.
    Right panel: horizontal bar chart of |corr| with objective.
    """
    n = len(params)
    ncols_hist = 4
    nrows_hist = int(np.ceil(n / ncols_hist))

    fig = plt.figure(figsize=(16, max(nrows_hist * 2.0, 6.0)))
    gs  = gridspec.GridSpec(nrows_hist, ncols_hist + 2,
                             figure=fig,
                             hspace=0.55, wspace=0.45)

    objs   = df["objective"].values
    corrs  = df[params].corrwith(df["objective"]).abs()

    for i, p in enumerate(params):
        r, c = divmod(i, ncols_hist)
        ax = fig.add_subplot(gs[r, c])
        ax.hist(df[p].values, bins=18, density=True,
                color=get_color(plot_cfg, "bo_valid", "#55A868"),
                alpha=0.75, edgecolor="none")
        ax.set_title(p, fontsize=7, pad=2)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)

    # Correlation bar chart in right columns
    ax_corr = fig.add_subplot(gs[:, ncols_hist:])
    sorted_corr = corrs.sort_values(ascending=True)
    bar_colors  = [
        get_color(plot_cfg, "bo_best",  "#DD8452")
        if v > sorted_corr.quantile(0.5)
        else get_color(plot_cfg, "bo_initial", "#4C72B0")
        for v in sorted_corr.values
    ]
    ax_corr.barh(range(len(sorted_corr)), sorted_corr.values,
                 color=bar_colors, edgecolor="none")
    ax_corr.set_yticks(range(len(sorted_corr)))
    ax_corr.set_yticklabels(sorted_corr.index.tolist(), fontsize=8)
    ax_corr.set_xlabel("|Pearson corr| with objective", fontsize=9)
    ax_corr.set_title("Param importance", fontsize=10)
    ax_corr.axvline(0.1, ls="--", lw=0.8,
                    color=get_color(plot_cfg, "refline", "#888888"))

    fig.suptitle(f"BO Parameter Space ({n} free params)", fontsize=13)

    data_df = df[params + ["objective"]].copy()
    data_df["corr_with_obj"] = df[params].corrwith(df["objective"]).reindex(
        data_df.columns.difference(["objective", "corr_with_obj"]),
        fill_value=np.nan).values[0] if False else np.nan
    # Simpler: just save param values + objective
    data_df = df[params + ["objective", "success"]].copy()
    save_fig(fig, "bo_param_space", figs_dir, plot_cfg, data_df)


# ============================================================================
# 3. Feasibility map
# ============================================================================

def plot_feasibility_map(bo_csv:   Union[str, Path],
                         plot_cfg: dict,
                         figs_dir: Union[str, Path]):
    """
    Visualise which parameter regions are feasible (LAMMPS succeeded).

    Selects the 2 params with the strongest feasibility signal (highest
    |point-biserial correlation| between param value and feasibility flag).
    Falls back to a 1-D feasibility rate bar chart when N=1.
    """
    df = pd.read_csv(bo_csv)
    df["feasible"] = df["success"].astype(bool)

    param_cols = detect_param_cols(df)
    n_params   = len(param_cols)

    if n_params == 0:
        print("    plot_feasibility_map: no param columns found")
        return

    # Compute |correlation| between each param and feasibility flag
    feas_int = df["feasible"].astype(int)
    corrs    = {p: abs(df[p].corr(feas_int)) for p in param_cols
                if df[p].notna().sum() > 5}

    n_feas   = df["feasible"].sum()
    n_total  = len(df)

    if n_params == 1 or len(corrs) < 2:
        _feasibility_1d(df, param_cols[0], plot_cfg, figs_dir,
                        n_feas, n_total)
        return

    # Pick top-2 params by |corr| with feasibility
    sorted_corr = sorted(corrs.items(), key=lambda kv: -kv[1])
    px, py      = sorted_corr[0][0], sorted_corr[1][0]

    fig, ax = plt.subplots(
        figsize=get_figsize(plot_cfg, "feasibility", [6.5, 5.0]))

    for label, mask, color, marker, zorder in [
        ("Infeasible", ~df["feasible"],
         get_color(plot_cfg, "infeasible",  "#C44E52"), "x", 1),
        ("Feasible",   df["feasible"],
         get_color(plot_cfg, "feasible",    "#55A868"), "o", 2),
    ]:
        ax.scatter(df.loc[mask, px], df.loc[mask, py],
                   s=18, alpha=0.55, c=color,
                   marker=marker, zorder=zorder, label=label)

    rate = 100.0 * n_feas / max(n_total, 1)
    ax.set_xlabel(px)
    ax.set_ylabel(py)
    ax.set_title(
        f"Feasibility Map  (feasible={n_feas}/{n_total}, {rate:.0f}%)\n"
        f"2 most informative params shown"
    )
    ax.legend(framealpha=0.85)

    data_df = df[[px, py, "feasible", "objective"]].copy()
    save_fig(fig, "feasibility_map", figs_dir, plot_cfg, data_df)


def _feasibility_1d(df:       pd.DataFrame,
                    param:    str,
                    plot_cfg: dict,
                    figs_dir: Union[str, Path],
                    n_feas:   int,
                    n_total:  int):
    """Single-parameter feasibility rate histogram (fallback for N=1)."""
    fig, ax = plt.subplots(
        figsize=get_figsize(plot_cfg, "feasibility", [6.5, 5.0]))

    bins = np.linspace(df[param].min(), df[param].max(), 15)
    ax.hist(df.loc[df["feasible"],  param], bins=bins, alpha=0.7,
            color=get_color(plot_cfg, "feasible",   "#55A868"),
            label="Feasible",   density=False)
    ax.hist(df.loc[~df["feasible"], param], bins=bins, alpha=0.7,
            color=get_color(plot_cfg, "infeasible", "#C44E52"),
            label="Infeasible", density=False)

    ax.set_xlabel(param)
    ax.set_ylabel("Count")
    rate = 100.0 * n_feas / max(n_total, 1)
    ax.set_title(f"Feasibility Distribution  ({rate:.0f}% feasible)")
    ax.legend()

    data_df = df[[param, "feasible"]].copy()
    save_fig(fig, "feasibility_map", figs_dir, plot_cfg, data_df)
