#!/usr/bin/env python3
"""
Force Field BO + NN Pipeline -- v8 entry point.

Modes
-----
  python run.py                            # full BO with configs/recipes/cluster_full.yaml
  python run.py --resume                   # resume BO from checkpoint
  python run.py --dry-run                  # single LAMMPS at best known params
  python run.py --dry-run --no-mpi         # same, serial (login/management nodes)
  python run.py --dry-run --save-traj      # also save NPT trajectory + structures
  python run.py --plot <bo_dir>            # generate all paper figures

--dry-run auto-loads parameters (priority: AL final → NN optimize → BO best).
--no-mpi : uses mpiexec -n 1 with I_MPI_FABRICS=shm.
           Requires Intel MPI module: module load mpi/2021.15
           Use on management/login nodes where InfiniBand/OFI is unavailable.

v8 changes vs v6:
  - load_config(): validates v8 schema (atom_types, pair_params, charge, etc.).
    Removed: parameters, fixed_params.  Added: atom_types, pair_params, charge,
    compute_surface, n_bo_iterations.
  - _load_best_params(): fully generic — parses any "name = value" pair from
    best_parameters.txt (not limited to ep_atom1 / si_atom1).
  - cmd_dry_run(): displays all free BO params dynamically (not hardcoded names).
  - ForceFieldOptimizer v8: auto-selects GP / TuRBO / SAASBO by n_params.

After BO completes:
  python slurm/submit.py nn --config configs/recipes/cluster_full.yaml
  python slurm/submit.py al --config configs/recipes/cluster_full.yaml
  python run.py --plot <bo_dir>
"""

import argparse
import glob as _glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .config_loader import load_config as load_expanded_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# Config loading and validation
# ============================================================================

def load_config(path: str) -> dict:
    """
    Load and validate config.yaml (v8 schema).

    All sections listed below are required — the pipeline fails fast if any
    are missing, avoiding hard-to-debug errors later in long BO runs.
    """
    config = load_expanded_config(path)

    # ── Required top-level sections ─────────────────────────────────────────
    required_sections = [
        "manifest", "atom_types", "pair_params", "charge",
        "targets", "output_mapping", "lammps", "parallel",
        "surf_energy_sanity", "optimization", "checkpoint",
        "nn", "active_learning",
    ]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    # ── manifest ─────────────────────────────────────────────────────────────
    required_manifest = ["system_name", "crystal_structure",
                         "surface_facet", "data_files"]
    miss_m = [k for k in required_manifest if k not in config["manifest"]]
    if miss_m:
        raise ValueError(f"manifest missing keys: {miss_m}")

    # ── data_files: bulk always required; surf files only when compute_surface ─
    data_files = config["manifest"]["data_files"]
    if "bulk" not in data_files:
        raise ValueError("manifest.data_files.bulk is required")
    if config["lammps"].get("compute_surface", True):
        for key in ("surf_complete", "surf_split"):
            if key not in data_files:
                raise ValueError(
                    f"manifest.data_files.{key} is required when "
                    "lammps.compute_surface: true"
                )

    # ── atom_types: at least one entry with required fields ──────────────────
    if not config["atom_types"]:
        raise ValueError("atom_types must have at least one entry")
    for i, at in enumerate(config["atom_types"]):
        for field in ("type", "label", "mass", "params"):
            if field not in at:
                raise ValueError(
                    f"atom_types[{i}] missing field '{field}'"
                )

    # ── targets: must be non-empty; surf_energy allowed only with compute_surface ─
    if not config["targets"]:
        raise ValueError("targets section must have at least one entry")
    if ("surf_energy" in config["targets"] and
            not config["lammps"].get("compute_surface", True)):
        print("  NOTE: 'surf_energy' target is defined but lammps.compute_surface "
              "is false — surf_energy will be excluded from the objective.")

    # ── lammps: required sub-keys ─────────────────────────────────────────────
    lmp = config["lammps"]
    for k in ("executable", "bulk_input", "surf_input",
              "compute_surface", "cutoff", "timestep", "timeout"):
        if k not in lmp:
            raise ValueError(f"lammps.{k} is required")
    for section, keys in [
        ("bulk", ("nx", "ny", "nz", "npt_seed", "equil_steps", "prod_steps")),
        ("surf", ("npt_seed", "equil_steps", "prod_steps")),
    ]:
        if section not in lmp:
            raise ValueError(f"lammps.{section} section is required")
        for k in keys:
            if k not in lmp[section]:
                raise ValueError(f"lammps.{section}.{k} is required")

    # ── optimization: required keys ───────────────────────────────────────────
    opt = config["optimization"]
    for k in ("method", "n_initial", "n_bo_iterations", "objective"):
        if k not in opt:
            raise ValueError(f"optimization.{k} is required")

    # ── output_mapping: column order must be consistent ───────────────────────
    bulk_map = config["output_mapping"].get("bulk", {})
    if "columns" not in bulk_map:
        raise ValueError("output_mapping.bulk.columns is required")

    return config


# ============================================================================
# Auto-loader: best parameters from pipeline results
# ============================================================================

def _load_best_params(config: dict = None):
    """
    Auto-detect best parameters from pipeline results.

    Priority
    --------
    1. AL final (active_learning_output/nn_round_final/nn_optimize_result.json)
    2. AL by round (newest active_learning_output/nn_round_*/nn_optimize_result.json)
    3. NN optimize (newest nn_output_<system>_*/nn_optimize_result.json)
    4. BO best_parameters.txt (newest bo_*/best_parameters.txt)

    Returns
    -------
    (params_dict, source_label) or (None, None) if nothing found.

    params_dict keys are dynamic (e.g. "Fe_corner_epsilon") — no hardcoding.
    """

    def _from_json(path: str) -> dict:
        """Parse nn_optimize_result.json; returns param dict or None."""
        try:
            with open(path) as f:
                d = json.load(f)
            # LAMMPS-validated result takes priority over NN-predicted
            best = d.get("best_lammps") or d.get("best", {})
            # Support both flat dict and nested "params" sub-dict
            params = best.get("params", best)
            if not params:
                return None
            # Filter: only numeric values that look like BO parameters
            result = {}
            for k, v in params.items():
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    pass
            return result if result else None
        except Exception:
            return None

    def _from_best_params_txt(path: str) -> dict:
        """
        Parse best_parameters.txt written by optimizer._save_summary_files.

        v8 format (one param per line):
            # comment lines starting with #
            Fe_corner_epsilon = 0.50123456  # range [0.01, 10.0]
            Fe_corner_sigma   = 2.34567890  # range [1.0, 4.0]
        Also handles v6 format:
            ep_atom1 = 0.5
            si_atom1 = 2.5
        """
        try:
            params = {}
            with open(path) as f:
                for line in f:
                    line = line.split("#")[0].strip()   # strip inline comments
                    if not line or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        params[k] = float(v)
                    except ValueError:
                        pass
            return params if params else None
        except Exception:
            return None

    sysname = (config or {}).get("manifest", {}).get("system_name", "")

    # 1. AL final -- prefer system-scoped output dirs.
    al_roots = []
    if sysname:
        al_roots.extend(sorted(
            _glob.glob(f"active_learning_output_{sysname}"),
            key=os.path.getmtime, reverse=True))
    al_roots.append("active_learning_output")
    for root in al_roots:
        al_final = os.path.join(root, "nn_round_final", "nn_optimize_result.json")
        p = _from_json(al_final)
        if p:
            return p, f"AL final  ({al_final})"

    # 2. AL by round (newest)
    al_jsons = []
    for root in al_roots:
        al_jsons.extend(_glob.glob(
            os.path.join(root, "nn_round_*", "nn_optimize_result.json")))
    al_jsons = sorted(al_jsons, key=os.path.getmtime, reverse=True)
    if al_jsons:
        p = _from_json(al_jsons[0])
        if p:
            return p, f"AL round  ({al_jsons[0]})"

    # 3. NN optimize -- prefer timestamped outputs matching THIS config.
    nn_jsons = []
    if sysname:
        nn_jsons = sorted(
            _glob.glob(f"nn_output_{sysname}_*/nn_optimize_result.json"),
            key=os.path.getmtime, reverse=True)
    if not nn_jsons:
        nn_jsons = sorted(
            _glob.glob("nn_output_*/nn_optimize_result.json"),
            key=os.path.getmtime, reverse=True)
    if nn_jsons:
        p = _from_json(nn_jsons[0])
        if p:
            return p, f"NN optimize  ({nn_jsons[0]})"
    p = _from_json("nn_output/nn_optimize_result.json")
    if p:
        return p, "NN optimize  (legacy nn_output/nn_optimize_result.json)"

    # 4. BO best_parameters.txt — prefer bo dirs matching THIS config's
    #    system_name (so cfg1/cfg2/full experiments load their OWN best),
    #    then fall back to the newest bo_* of any name.
    bo_txts = []
    if sysname:
        bo_txts = sorted(_glob.glob(f"bo_{sysname}_*/best_parameters.txt"),
                         key=os.path.getmtime, reverse=True)
    if not bo_txts:
        bo_txts = sorted(_glob.glob("bo_*/best_parameters.txt"),
                         key=os.path.getmtime, reverse=True)
    if bo_txts:
        p = _from_best_params_txt(bo_txts[0])
        if p:
            return p, f"BO best  ({bo_txts[0]})"

    return None, None


# ============================================================================
# Commands
# ============================================================================

def cmd_dry_run(config: dict,
                config_path: str = "configs/recipes/cluster_full.yaml",
                no_mpi: bool = False,
                save_traj: bool = False) -> bool:
    """
    Run a single LAMMPS evaluation to validate the pipeline wiring.

    Parameters are auto-loaded from AL / NN / BO results (priority order).
    All free BO parameter names are printed dynamically — no hardcoded names.

    --no-mpi   : override use_mpi=true; needed on login/management nodes
    --save-traj: write NPT trajectory + final structure files
    """
    print()
    print("=" * 70)
    print("DRY RUN: single LAMMPS evaluation")
    print("=" * 70)
    print()

    # ── Build expected free param names from config ──────────────────────────
    from .optimizer import ForceFieldOptimizer
    param_space = ForceFieldOptimizer._build_param_space(config)
    param_names = [p[0] for p in param_space]

    print(f"  Free parameters ({len(param_names)}):")
    for name, lo, hi in param_space:
        print(f"    {name:<28} [{lo}, {hi}]")
    print()

    # ── Auto-load best parameters ─────────────────────────────────────────────
    ref, source = _load_best_params(config)
    if ref:
        # Filter to only the params relevant to this config's parameter space
        matched = {k: v for k, v in ref.items() if k in param_names}
        missing_keys = [k for k in param_names if k not in matched]

        print(f"  Auto-loaded from: {source}")
        for k, v in matched.items():
            print(f"    {k:<28} = {v:.8f}")

        if missing_keys:
            print(f"\n  WARNING: The following params were not found in the "
                  f"loaded file and will use midpoint defaults:")
            for k in missing_keys:
                lo = next(p[1] for p in param_space if p[0] == k)
                hi = next(p[2] for p in param_space if p[0] == k)
                mid = (lo + hi) / 2.0
                matched[k] = mid
                print(f"    {k:<28} = {mid:.6f}  (midpoint [{lo}, {hi}])")
        ref = matched
    else:
        # Fallback: midpoint of each parameter range
        print("  WARNING: no pipeline results found; using midpoint defaults")
        ref = {}
        for name, lo, hi in param_space:
            mid = (lo + hi) / 2.0
            ref[name] = mid
            print(f"    {name:<28} = {mid:.6f}  (midpoint [{lo}, {hi}])")
    print()

    # ── Apply --no-mpi override ───────────────────────────────────────────────
    if no_mpi:
        config["parallel"]["use_mpi"] = False
        config["parallel"]["cores_per_worker"] = 1
        print("  --no-mpi: overriding use_mpi=false, cores_per_worker=1")
        print("  I_MPI_FABRICS=shm will be set automatically (shared-memory MPI)")
        print()

    # ── Run evaluation ────────────────────────────────────────────────────────
    from .lammps_interface import LAMMPSRunner
    runner   = LAMMPSRunner(config)
    work_dir = os.path.abspath("dry_run")
    os.makedirs(work_dir, exist_ok=True)

    print(f"  Work dir  : {work_dir}")
    print(f"  save_traj : {save_traj}")
    print(f"  Running LAMMPS...")
    print()

    results = runner.evaluate_batch([ref], work_dir, save_traj=save_traj)
    r = results[0]

    print()
    print("=" * 70)
    print("DRY RUN RESULT")
    print("=" * 70)
    print(f"  success   : {r.success}")
    print(f"  objective : {r.objective:.6f}" if r.success else
          f"  error     : {r.error_msg}")

    if not r.success:
        print()
        print("  Check eval_0000/{bulk,surf_complete,surf_split}/lammps_stdout.log")
        return False

    print()
    print("  Computed properties:")
    targets = config["targets"]
    compute_surf = config["lammps"]["compute_surface"]
    for k, v in r.properties.items():
        if k in targets:
            tgt = float(targets[k]["value"])
            dev = (v - tgt) / tgt * 100.0
            unit = targets[k].get("unit", "")
            print(f"    {k:<16} = {v:>12.4f} {unit:<8}  "
                  f"(target {tgt}, dev {dev:+.2f}%)")
        elif k not in ("E_complete", "E_split", "A_xy", "a_0K"):
            # Print auxiliary props without target comparison
            print(f"    {k:<16} = {v:>12.4f}")

    if "surf_energy" in r.properties and not compute_surf:
        # Shouldn't happen, but guard
        pass

    print()
    print("  Per-property errors (%):")
    for k, v in r.per_property_error.items():
        flag = "  ← !" if v > 5.0 else ""
        print(f"    {k:<16} = {v:>8.2f}%{flag}")

    print()
    if r.success:
        print("  Pipeline wiring: OK")
        print(f"  Current config validated: {config_path}")
        print("  To launch production BO, use a production recipe such as "
              "configs/recipes/local_full.yaml or configs/recipes/cluster_full.yaml")
    return True


def _reference_params(config: dict) -> dict:
    """Load best known params for this config, falling back to range midpoints."""
    from .optimizer import ForceFieldOptimizer

    param_space = ForceFieldOptimizer._build_param_space(config)
    param_names = [p[0] for p in param_space]
    ref, source = _load_best_params(config)
    if ref:
        matched = {k: v for k, v in ref.items() if k in param_names}
        for name, lo, hi in param_space:
            if name not in matched:
                matched[name] = (lo + hi) / 2.0
        print(f"  Params      : {source}")
        return matched

    print("  Params      : midpoint defaults (no BO/NN/AL results found)")
    return {name: (lo + hi) / 2.0 for name, lo, hi in param_space}


def cmd_sublimation_test(config: dict,
                         config_path: str = "configs/recipes/cluster_full.yaml",
                         no_mpi: bool = False,
                         save_traj: bool = False) -> bool:
    """Run the NPT-bulk sublimation-energy proxy audit for one parameter set."""
    print()
    print("=" * 70)
    print("SUBLIMATION PROXY TEST")
    print("=" * 70)

    if not config.get("sublimation", {}).get("enabled", False):
        print("  ERROR: sublimation.enabled is false in config")
        return False

    if no_mpi:
        config["parallel"]["use_mpi"] = False
        config["parallel"]["cores_per_worker"] = 1
        print("  --no-mpi: overriding use_mpi=false, cores_per_worker=1")

    params = _reference_params(config)

    from .lammps_interface import LAMMPSRunner
    runner = LAMMPSRunner(config)
    work_dir = os.path.abspath("sublimation_test")
    print(f"  Config      : {config_path}")
    print(f"  Work dir    : {work_dir}")
    print("  Running bulk NPT + single-molecule minimisation...")

    results = runner.evaluate_batch([params], work_dir, save_traj=save_traj)
    r = results[0]
    print()
    print("=" * 70)
    print("SUBLIMATION RESULT")
    print("=" * 70)
    if not r.success:
        print("  success     : False")
        print(f"  error       : {r.error_msg}")
        print("  Check sublimation_test/eval_0000/{bulk,sublimation/sub_single}/lammps_stdout.log")
        return False

    props = r.properties
    target = config["targets"].get("esub_proxy", {})
    sub_cfg = config["sublimation"]
    default_proxy_target = (
        float(sub_cfg.get("target_kj_mol", float("nan")))
        - float(sub_cfg.get("thermal_correction_kj_mol", 0.0))
    )
    target_proxy = float(target.get("value", default_proxy_target))
    esub_kj = props["esub_proxy"]
    esub_kcal = props.get("esub_proxy_kcal_mol", esub_kj / 4.184)

    print("  success     : True")
    print(f"  E_bulk NPT  : {props['pe']:.6f} kcal/mol-box")
    print(f"  H_bulk NPT  : {props['enthalpy']:.6f} kcal/mol-box")
    print(f"  E_bulk/Nmol : {props['esub_bulk_pe_per_mol']:.6f} kcal/mol")
    print(f"  E_single    : {props['esub_single_pe']:.6f} kcal/mol")
    print(f"  esub_proxy  : {esub_kcal:.6f} kcal/mol")
    print(f"              : {esub_kj:.6f} kJ/mol")
    print(f"  proxy target: {target_proxy:.6f} kJ/mol")
    print(f"  error       : {esub_kj - target_proxy:+.6f} kJ/mol")
    return True


def cmd_plot(config: dict, bo_dir: str, nn_dir: str = None) -> bool:
    """Generate all paper figures from a completed BO run."""
    from viz import (load_plot_config, apply_style,
                     plot_bo_convergence, plot_bo_param_space,
                     plot_feasibility_map,
                     plot_nn_parity, plot_nn_optimize_result,
                     plot_optimize_candidates, plot_lammps_validation,
                     plot_active_learning_curve)

    bo_dir = Path(bo_dir)
    bo_csv = bo_dir / "all_results.csv"
    if not bo_csv.exists():
        print(f"ERROR: {bo_csv} not found")
        return False

    plot_cfg  = load_plot_config("configs/plot.yaml")
    apply_style(plot_cfg)
    figs_dir  = Path("figures")
    figs_dir.mkdir(exist_ok=True)

    print(f"Generating figures from {bo_dir} → {figs_dir}/")

    print("  [1/8] bo_convergence")
    plot_bo_convergence(bo_csv, plot_cfg, figs_dir)

    print("  [2/8] bo_param_space")
    plot_bo_param_space(bo_csv, plot_cfg, figs_dir)

    print("  [3/8] feasibility_map (SI)")
    plot_feasibility_map(bo_csv, plot_cfg, figs_dir)

    # NN figures: prefer --nn-dir, else auto-select latest nn_output_*
    if nn_dir:
        nn_dir = Path(nn_dir)
    else:
        candidates = sorted(
            _glob.glob("nn_output_*"), key=os.path.getmtime, reverse=True
        )
        nn_dir = Path(candidates[0]) if candidates else Path("nn_output")
    print(f"  NN dir: {nn_dir}")

    if (nn_dir / "forward_nn.pt").exists():
        print("  [4/8] nn_parity")
        plot_nn_parity(bo_csv, nn_dir / "forward_nn.pt", plot_cfg, figs_dir,
                       config=config)
    else:
        print("  [4/8] nn_parity -- SKIP (forward_nn.pt missing)")

    opt_json = nn_dir / "nn_optimize_result.json"
    if opt_json.exists():
        print("  [5/8] nn_optimize_result")
        plot_nn_optimize_result(opt_json, plot_cfg, figs_dir, bo_csv_path=bo_csv)

        print("  [6/8] optimize_candidates")
        plot_optimize_candidates(opt_json, plot_cfg, figs_dir)

        print("  [7/8] lammps_validation")
        plot_lammps_validation(opt_json, plot_cfg, figs_dir)
    else:
        print("  [5-7/8] optimize + candidates + validation -- "
              "SKIP (nn_optimize_result.json missing)")

    sysname = config.get("manifest", {}).get("system_name", "")
    al_candidates = []
    if sysname:
        al_candidates.append(
            Path(f"active_learning_output_{sysname}") / "active_learning_history.json")
    al_candidates.append(Path("active_learning_output") / "active_learning_history.json")
    al_json = next((p for p in al_candidates if p.exists()), al_candidates[-1])
    if al_json.exists():
        print("  [8/8] active_learning_curve")
        plot_active_learning_curve(al_json, plot_cfg, figs_dir)
    else:
        print("  [8/8] active_learning_curve -- SKIP (no AL run yet)")

    print()
    print(f"Done. Figures in {figs_dir}/")
    print("  Each figure: .pdf (vector), .png (raster), _data.csv (raw data)")
    return True


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Force Field BO + NN Pipeline v8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/recipes/cluster_full.yaml",
                        help="Path to config YAML or modular recipe.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume BO from latest checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to specific checkpoint JSON to resume from")
    parser.add_argument("--bo-method",
                        choices=["auto", "gp", "turbo", "saasbo"],
                        default=None,
                        help="Override optimization.method without editing config.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single LAMMPS evaluation (pipeline validation). "
                             "Params auto-loaded from AL → NN → BO results.")
    parser.add_argument("--sublimation-test", action="store_true",
                        help="Run an NPT-bulk sublimation/cohesive-energy proxy audit.")
    parser.add_argument("--no-mpi", action="store_true",
                        help="Override config use_mpi=true; run serial LAMMPS "
                             "(mpiexec -n 1 + I_MPI_FABRICS=shm). "
                             "Required on login/management nodes without "
                             "InfiniBand/OFI fabric.")
    parser.add_argument("--save-traj", action="store_true",
                        help="(--dry-run only) Save NPT trajectory + final "
                             "structure files for post-processing.")
    parser.add_argument("--plot", metavar="BO_DIR", default=None,
                        help="Generate all paper figures from a completed BO "
                             "run directory (must contain all_results.csv)")
    parser.add_argument("--nn-dir", default=None,
                        help="NN output dir for --plot (default: auto-select "
                             "latest nn_output_*/)")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("#" * 70)
    print("# Force Field BO + NN Pipeline v8")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    # ── Load and validate config ───────────────────────────────────────────────
    print(f"\nLoading config: {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)
    except ValueError as e:
        print(f"Config validation error: {e}")
        sys.exit(1)

    if args.bo_method:
        config["optimization"]["method"] = args.bo_method

    mf = config["manifest"]
    lmp = config["lammps"]
    print(f"  system      : {mf['system_name']}")
    print(f"  structure   : {mf.get('crystal_structure', 'N/A')}")
    print(f"  facet       : {mf.get('surface_facet', 'N/A')}")
    print(f"  atom types  : {len(config['atom_types'])} "
          f"({', '.join(at['label'] for at in config['atom_types'])})")
    print(f"  charge      : {'ON' if config['charge']['enabled'] else 'OFF'}")
    print(f"  surf energy : {'ON' if lmp['compute_surface'] else 'OFF'}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.dry_run:
        ok = cmd_dry_run(config, config_path=args.config,
                         no_mpi=args.no_mpi, save_traj=args.save_traj)
        sys.exit(0 if ok else 1)

    if args.sublimation_test:
        ok = cmd_sublimation_test(config, config_path=args.config,
                                  no_mpi=args.no_mpi,
                                  save_traj=args.save_traj)
        sys.exit(0 if ok else 1)

    if args.plot:
        ok = cmd_plot(config, args.plot, nn_dir=args.nn_dir)
        sys.exit(0 if ok else 1)

    # ── Full BO ───────────────────────────────────────────────────────────────
    from .optimizer import ForceFieldOptimizer
    optimizer = ForceFieldOptimizer(config)

    if args.resume or args.checkpoint:
        ok = optimizer.load_checkpoint(args.checkpoint)
        if not ok and args.resume:
            print("No checkpoint found; starting fresh BO run")

    try:
        summary = optimizer.run()
        if summary:
            sys.name = mf["system_name"]
            print(f"\n{'='*60}")
            print("BO completed successfully.")
            print(f"{'='*60}")
            machine = "cluster" if "cluster" in config else "local"
            print("\nNext steps:")
            print(f"  1. python ffopt.py nn --machine {machine}")
            print(f"  2. python ffopt.py al --machine {machine}")
            print("  3. python ffopt.py plot")
            print(f"  4. python ffopt.py validate --machine {machine}")
        else:
            print("\nBO failed — no valid results found.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving checkpoint...")
        optimizer._save_checkpoint()
        machine = "cluster" if "cluster" in config else "local"
        print(f"Checkpoint saved. Resume with: python ffopt.py bo --machine {machine} --resume")
        sys.exit(130)


if __name__ == "__main__":
    main()
