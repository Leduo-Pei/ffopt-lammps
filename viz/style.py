"""
viz/style.py -- Styling helpers for all pipeline figures (v8)

Public API
----------
  load_plot_config(path)  : load plot_config.yaml → dict
  apply_style(plot_cfg)   : set matplotlib rcParams from plot_cfg
  save_fig(fig, stem, figs_dir, plot_cfg)  : save .pdf + .png
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ── Fixed column names in all_results.csv (not free BO parameters) ───────────
# Used by bo_plots to detect param columns dynamically.
_FIXED_COLS = frozenset({
    "round", "label", "success", "objective", "error_msg",
})


def load_plot_config(path: str | None = None) -> dict:
    """
    Load plot_config.yaml.

    Returns an empty dict (with sensible defaults applied elsewhere) when the
    file does not exist — this allows the pipeline to produce figures even
    without a plot_config.yaml in the working directory.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"  plot config not found at {path}; using defaults")
        return {}
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def apply_style(plot_cfg: dict):
    """
    Apply matplotlib rcParams from plot_config.yaml.

    Called once at the start of run.py --plot before any figures are drawn.
    Idempotent — safe to call multiple times.
    """
    font_cfg  = plot_cfg.get("font",  {})
    axes_cfg  = plot_cfg.get("axes",  {})

    rc = {
        "font.family":        font_cfg.get("family",         "DejaVu Sans"),
        "font.size":          font_cfg.get("size",           11),
        "axes.labelsize":     font_cfg.get("axes_labelsize", 12),
        "xtick.labelsize":    font_cfg.get("tick_labelsize", 10),
        "ytick.labelsize":    font_cfg.get("tick_labelsize", 10),
        "legend.fontsize":    font_cfg.get("legend_fontsize", 10),
        "axes.titlesize":     font_cfg.get("title_fontsize", 13),
        "axes.grid":          axes_cfg.get("grid",           True),
        "grid.alpha":         axes_cfg.get("grid_alpha",     0.3),
        "axes.spines.top":    not axes_cfg.get("spine_top",  False),
        "axes.spines.right":  not axes_cfg.get("spine_right", False),
        "figure.dpi":         plot_cfg.get("export", {}).get("dpi", 150),
        "savefig.dpi":        plot_cfg.get("export", {}).get("dpi", 300),
        "savefig.bbox_inches": plot_cfg.get("export", {}).get("bbox_inches", "tight"),
    }
    plt.rcParams.update(rc)


def save_fig(fig: plt.Figure,
             stem:     str,
             figs_dir: Union[str, Path],
             plot_cfg: dict,
             data_df:  Optional[pd.DataFrame] = None):
    """
    Save figure as .pdf and .png, and optionally a _data.csv alongside.

    Parameters
    ----------
    fig      : matplotlib Figure to save
    stem     : base filename without extension (e.g. "bo_convergence")
    figs_dir : output directory
    plot_cfg : parsed plot_config.yaml dict
    data_df  : optional DataFrame to save as {stem}_data.csv
    """
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = plot_cfg.get("export", {})
    dpi        = export_cfg.get("dpi",         300)
    bbox       = export_cfg.get("bbox_inches", "tight")
    formats    = export_cfg.get("formats",     ["pdf", "png"])

    for fmt in formats:
        out_path = figs_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi if fmt == "png" else None,
                    bbox_inches=bbox)

    if data_df is not None:
        csv_path = figs_dir / f"{stem}_data.csv"
        data_df.to_csv(csv_path, index=False)

    plt.close(fig)


def get_color(plot_cfg: dict, key: str, default: str = "#4C72B0") -> str:
    """Look up a color from plot_cfg['colors'] with a fallback default."""
    return plot_cfg.get("colors", {}).get(key, default)


def get_figsize(plot_cfg: dict,
                key:     str,
                default: List[float] = None) -> List[float]:
    """Look up a figure size from plot_cfg['figure_size']."""
    if default is None:
        default = [8.0, 5.0]
    return plot_cfg.get("figure_size", {}).get(key, default)


def detect_param_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the free BO parameter column names from an all_results.csv DataFrame.

    Strategy: exclude fixed metadata columns and any column whose name starts
    with "calc_" or "error_".  Whatever remains are the BO param columns.

    Works for any number of atom types and any mixing-rule configuration
    (including cross_1_2_epsilon / cross_1_2_sigma columns).
    """
    return [
        c for c in df.columns
        if c not in _FIXED_COLS
        and not c.startswith("calc_")
        and not c.startswith("error_")
    ]
