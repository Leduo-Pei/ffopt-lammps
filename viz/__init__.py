"""
viz -- Force Field BO + NN Pipeline Visualisation Package (v8)

Public API (consumed by run.py --plot via  `from viz import ...`)
----------------------------------------------------------------
  load_plot_config(path)
      Load plot_config.yaml → dict.  Returns empty dict if file not found.

  apply_style(plot_cfg)
      Apply matplotlib rcParams from plot_config dict.

  plot_bo_convergence(bo_csv, plot_cfg, figs_dir)
      Best-objective convergence curve.

  plot_bo_param_space(bo_csv, plot_cfg, figs_dir)
      Parameter space pair-plot (adaptive: full / top-6 / marginals).
      Fully dynamic — detects param column names from CSV automatically.

  plot_feasibility_map(bo_csv, plot_cfg, figs_dir)
      2-D feasibility scatter of the two most informative parameters.

  plot_nn_parity(bo_csv, pt_path, plot_cfg, figs_dir, config=None)
      NN ensemble predicted vs. actual LAMMPS objective (parity plot).

  plot_nn_optimize_result(opt_json, plot_cfg, figs_dir, bo_csv_path=None)
      Per-property error bar chart for the NN-optimal parameter set.

  plot_optimize_candidates(opt_json, plot_cfg, figs_dir)
      All NN-optimization candidates: predicted objective vs. rank.

  plot_lammps_validation(opt_json, plot_cfg, figs_dir)
      Side-by-side LAMMPS vs. experiment property comparison.

  plot_active_learning_curve(al_json, plot_cfg, figs_dir)
      AL convergence: uncertainty + best objective + R² across rounds.

Each plot function saves  {name}.pdf,  {name}.png,  {name}_data.csv
in the supplied figs_dir.

v8 changes vs v6
----------------
  - plot_bo_param_space: dynamic param detection; adapts to N=1…17+ params
  - plot_nn_parity: loads EnsembleMLP via load_ensemble_from_file();
    shows ±1 ensemble std error bars (epistemic uncertainty)
  - plot_nn_optimize_result / plot_lammps_validation: use new
    nn_optimize_result.json schema (best_lammps key)
  - plot_active_learning_curve: dual-axis (uncertainty + R²); reads
    active_learning_history.json schema from active_learning.py v8
  - All functions: save _data.csv alongside every figure
"""

from .style import load_plot_config, apply_style

from .bo_plots import (
    plot_bo_convergence,
    plot_bo_param_space,
    plot_feasibility_map,
)

from .nn_plots import (
    plot_nn_parity,
    plot_nn_optimize_result,
    plot_optimize_candidates,
    plot_lammps_validation,
)

from .al_plots import plot_active_learning_curve

__all__ = [
    # Style
    "load_plot_config",
    "apply_style",
    # BO plots
    "plot_bo_convergence",
    "plot_bo_param_space",
    "plot_feasibility_map",
    # NN plots
    "plot_nn_parity",
    "plot_nn_optimize_result",
    "plot_optimize_candidates",
    "plot_lammps_validation",
    # AL plot
    "plot_active_learning_curve",
]
