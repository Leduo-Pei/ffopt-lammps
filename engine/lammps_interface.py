"""
lammps_interface.py  |  LAMMPS runner for force field BO  |  v8

Workflow per evaluation:
  1. Bulk tri NPT 300 K -> a_0K, a, b, c, alpha, beta, gamma_ang, density, pe, enthalpy
  2. Surf NVT 300 K (complete + split run in parallel) -> surf_energy
        # Stages 2+3: surf_complete and surf_split run in parallel.          #

pair_coeffs.lmp is written before each LAMMPS call (never passed as -var).
LAMMPS scripts must contain:  include pair_coeffs.lmp

EvalResult carries:
  objective       - scalar weighted RMSE for BO acquisition
  obj_structural  - weighted RMSE of non-surface targets
  obj_surface     - surface-energy percent error; nan when surface is off
"""

import os
import re
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .property_evaluators import PropertyEvaluationContext, build_property_plan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Penalty objective returned for failed or non-physical evaluations
LARGE_PENALTY = 1000.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Structured result for one BO evaluation point."""
    params:             Dict[str, float]
    properties:         Dict[str, float]
    objective:          float                # scalar weighted RMSE (BO acquisition)
    success:            bool
    error_msg:          str               = ""
    per_property_error: Dict[str, float]  = field(default_factory=dict)
    # Pareto group objectives (nan when surface not computed or eval failed)
    obj_structural:     float             = float("nan")  # RMSE of non-surf targets
    obj_surface:        float             = float("nan")  # surf_energy % error


# ---------------------------------------------------------------------------
# LAMMPSRunner
# ---------------------------------------------------------------------------

class LAMMPSRunner:
    """
    Multi-stage LAMMPS workflow for force field BO.

    Usage
    -----
    runner = LAMMPSRunner(config)
    results = runner.evaluate_batch(param_list, work_dir)

    Parameters
    ----------
    config : dict
        Parsed config.yaml (v8 schema).
    """

    def __init__(self, config: dict):
        # ------------------------------------------------------------------ #
        # Unpack config sections (all required; no hidden defaults)
        # ------------------------------------------------------------------ #
        lmp_cfg  = config["lammps"]
        manifest = config["manifest"]
        parallel = config["parallel"]
        pair_cfg = config["pair_params"]
        chg_cfg  = config["charge"]
        out_map  = config["output_mapping"]
        se_san   = config["surf_energy_sanity"]

        # -- executables and input scripts --
        self.lammps_exe  = lmp_cfg["executable"]
        self.mpiexec     = lmp_cfg.get("mpiexec", "mpiexec")
        self.bulk_input  = os.path.abspath(lmp_cfg["bulk_input"])
        self.surf_input  = os.path.abspath(lmp_cfg["surf_input"])

        # -- feature switches --
        self.compute_surface = lmp_cfg["compute_surface"]   # [bool]
        self.use_charge      = chg_cfg["enabled"]           # [bool]

        # -- pair parameter config --
        self.mixing_rule        = pair_cfg["mixing_rule"]
        self.derived_params_cfg = pair_cfg.get("derived_params", [])  # list of constraint dicts
        self.explicit_pairs_cfg = pair_cfg.get("explicit_pairs", [])  # list of cross-pair dicts

        # -- atom types --
        self.atom_types = config["atom_types"]   # list of {type, label, mass, params}

        # -- charge config --
        self.charge_cfg = chg_cfg

        # -- data files --
        self.bulk_data = os.path.abspath(manifest["data_files"]["bulk"])
        if self.compute_surface:
            self.surf_complete = os.path.abspath(manifest["data_files"]["surf_complete"])
            self.surf_split    = os.path.abspath(manifest["data_files"]["surf_split"])
        else:
            self.surf_complete = None
            self.surf_split    = None

        # -- MD simulation parameters --
        bulk_md = lmp_cfg["bulk"]
        surf_md = lmp_cfg["surf"]

        self.bulk_nx       = bulk_md["nx"]
        self.bulk_ny       = bulk_md["ny"]
        self.bulk_nz       = bulk_md["nz"]
        self.bulk_npt_seed = bulk_md["npt_seed"]
        self.bulk_equil    = bulk_md["equil_steps"]   # NPT equilibration steps (discarded)
        self.bulk_prod     = bulk_md["prod_steps"]    # NPT production steps (time-averaged)
        self.bulk_temperature = float(bulk_md.get("temperature", 300.0))
        self.bulk_pressure = float(bulk_md.get("pressure", 1.0))
        self.timestep      = lmp_cfg["timestep"]
        self.cutoff        = lmp_cfg["cutoff"]
        self.timeout       = lmp_cfg["timeout"]

        # Surf seed must equal bulk seed for thermodynamic consistency (v7 requirement).
        self.surf_npt_seed = surf_md["npt_seed"]
        self.surf_equil    = surf_md["equil_steps"]   # NVT equilibration steps (discarded)
        self.surf_prod     = surf_md["prod_steps"]    # NVT production steps (time-averaged)

        # -- parallel execution --
        self.cores       = parallel["cores_per_worker"]
        self.omp_threads = parallel.get("omp_threads_per_worker", 1)
        self.use_mpi     = parallel["use_mpi"]
        self.max_workers = max(1, int(parallel["max_workers"]))
        self.scheduler_launcher = parallel.get("scheduler_launcher")
        self.scheduler_node_count = max(1, int(parallel.get("scheduler_nodes", 1)))
        self.workers_per_node = max(1, int(parallel.get("workers_per_node", 1)))

        # -- targets (for objective and sanity references) --
        self.targets = config["targets"]

        # -- output mapping --
        bulk_map          = out_map["bulk"]
        self.bulk_col_map = bulk_map["columns"]           # ordered list of column names
        self.bulk_targets = bulk_map["target_keys"]       # {target_name: column_name}
        self.surf_E_key   = out_map["surf"]["E_slab_key"]
        self.surf_A_key   = out_map["surf"]["A_xy_key"]

        # -- surface energy sanity bounds --
        self.surf_energy_min = se_san["min"]
        self.surf_energy_max = se_san["max"]

        # -- sanity reference values from targets section --
        # Per-parameter refs so anisotropic cells (a != b != c, e.g. molecular
        # crystals) are gated each against their OWN target, not a single one.
        self.ref_lat     = {lp: self.targets.get(lp, {}).get("value", None)
                            for lp in ("a", "b", "c")}
        self.ref_a       = self.targets.get("a",       {}).get("value", None)
        self.ref_density = self.targets.get("density", {}).get("value", None)

        # -- kspace accuracy (for charged systems) --
        self.kspace_accuracy = chg_cfg.get("kspace_accuracy", 1.0e-5)

        # -- adsorption energy (3-box) target: E_ad = E_complex - E_slab - E_mol --
        # Au is the fixed (uncharged) metal; BTAH carries the optimised params.
        # E_slab is independent of the BTAH params -> computed once and cached.
        ads_cfg = config.get("adsorption", {})
        self.ads_enabled = ads_cfg.get("enabled", False)
        self._slab_pe = None
        if self.ads_enabled:
            adf = ads_cfg["data_files"]
            adi = ads_cfg["inputs"]
            adm = ads_cfg["md"]
            self.ads_complex       = os.path.abspath(adf["complex"])
            self.ads_slab          = os.path.abspath(adf["slab"])
            self.ads_mol           = os.path.abspath(adf["mol"])
            self.ads_complex_input = os.path.abspath(adi["complex"])
            self.ads_slab_input    = os.path.abspath(adi["slab"])
            self.ads_mol_input     = os.path.abspath(adi["mol"])
            self.metal_label = ads_cfg["metal"]["label"]
            self.ads_cutoff   = adm["cutoff"]
            self.ads_timestep = adm["timestep"]
            self.ads_temp     = adm["temp"]
            self.ads_seed     = adm["seed"]
            self.ads_equil    = adm["equil_steps"]
            self.ads_prod     = adm["prod_steps"]
            self.ads_kspace   = adm.get("kspace_accuracy", self.kspace_accuracy)

        # -- sublimation proxy target: E_single - E_bulk_per_molecule --
        # Main evaluations use the finite-temperature NPT mean bulk PE already
        # produced by in.bulk.mol; the standalone public helper below still
        # supports a deterministic 0 K bulk audit.
        sub_cfg = config.get("sublimation", {})
        self.sub_enabled = sub_cfg.get("enabled", False)
        if self.sub_enabled:
            sdf = sub_cfg["data_files"]
            sui = sub_cfg["inputs"]
            self.sub_bulk_data    = os.path.abspath(sdf.get("bulk", self.bulk_data))
            self.sub_single_data  = os.path.abspath(sdf["single"])
            self.sub_bulk_input   = os.path.abspath(sui["bulk"])
            self.sub_single_input = os.path.abspath(sui["single"])
            self.sub_molecule_atoms = int(sub_cfg.get("molecule_atoms", 14))
            self.sub_target_kj = float(sub_cfg.get("target_kj_mol", float("nan")))
            self.sub_thermal_correction_kj = float(
                sub_cfg.get("thermal_correction_kj_mol", 0.0))
            self.sub_cutoff = float(sub_cfg.get("cutoff", self.cutoff))
            self.sub_kspace = float(sub_cfg.get("kspace_accuracy",
                                                self.kspace_accuracy))

        # -- property execution dependencies --
        # Bulk NPT is needed only when a target is a bulk observable, or when
        # the NPT-PE sublimation proxy is enabled. Interface-only recipes can
        # therefore evaluate adsorption without paying for bulk NPT.
        bulk_output_names = (
            set(self.bulk_col_map)
            | set(self.bulk_targets.keys())
            | set(self.bulk_targets.values())
        )
        self.bulk_required = (
            self.sub_enabled
            or any(prop in bulk_output_names for prop in self.targets)
        )

        # -- precount atoms per type in bulk data (needed for charge neutrality) --
        self.bulk_type_counts = self._read_type_counts(self.bulk_data)

        # ------------------------------------------------------------------ #
        # Validate file paths                                                 #
        # ------------------------------------------------------------------ #
        required_paths = [
            (self.bulk_data, "bulk data file"),
        ]
        if self.bulk_required:
            required_paths.append((self.bulk_input, "bulk input script"))
        if self.ads_enabled:
            required_paths += [
                (self.ads_complex_input, "adsorption complex input script"),
                (self.ads_slab_input,    "adsorption slab input script"),
                (self.ads_mol_input,     "adsorption mol input script"),
                (self.ads_complex,       "adsorption complex data file"),
                (self.ads_slab,          "adsorption slab data file"),
                (self.ads_mol,           "adsorption mol data file"),
            ]
        if self.sub_enabled:
            required_paths += [
                (self.sub_bulk_input,   "sublimation bulk input script"),
                (self.sub_single_input, "sublimation single input script"),
                (self.sub_bulk_data,    "sublimation bulk data file"),
                (self.sub_single_data,  "sublimation single data file"),
            ]
        if self.compute_surface:
            required_paths += [
                (self.surf_input,    "surf input script"),
                (self.surf_complete, "surf_complete data file"),
                (self.surf_split,    "surf_split data file"),
            ]
        for path, label in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        # Cleavage method sanity: both slabs must have the same atom count.
        if self.compute_surface:
            n_c = self._read_atom_count(self.surf_complete)
            n_s = self._read_atom_count(self.surf_split)
            if n_c is not None and n_s is not None and n_c != n_s:
                raise ValueError(
                    f"Cleavage method requires equal atom counts in surf_complete "
                    f"({n_c}) and surf_split ({n_s}). Check data file generation."
                )

        self.property_plan = build_property_plan(config, self)

    # ====================================================================== #
    # Public API                                                              #
    # ====================================================================== #

    def evaluate_batch(self,
                       param_batch: List[Dict[str, float]],
                       work_dir: str,
                       save_traj: bool = False) -> List[EvalResult]:
        """
        Evaluate a batch of parameter sets in parallel worker processes.

        Parameters
        ----------
        param_batch : list of dicts
            Each dict maps BO parameter names to float values.
            Key format: "{label}_{param}", e.g. "Fe_corner_epsilon".
        work_dir : str
            Parent directory; each evaluation gets its own eval_NNNN/ subdir.
        save_traj : bool
            If True, LAMMPS writes trajectory + final structure files.
            Only use for final validation runs, never for BO/AL batches.

        Returns
        -------
        list of EvalResult, same length and order as param_batch.
        """
        results: List[Optional[EvalResult]] = [None] * len(param_batch)
        # E_slab depends only on the (fixed) Au LJ -> compute ONCE in the main
        # process and cache; the value is then pickled to every worker.
        if self.ads_enabled and self._slab_pe is None:
            self._slab_pe = self._run_ads_box(
                {}, work_dir, "_slab_ref", self.ads_slab, self.ads_slab_input,
                has_charge=False, save_traj=False)
            print(f"  [adsorption] reference E_slab = {self._slab_pe} kcal/mol")

        worker_count = min(self.max_workers, len(param_batch))
        executor_cls = ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor
        executor_name = "threads" if os.name == "nt" else "processes"
        print(
            f"  LAMMPS batch workers: {worker_count}/{len(param_batch)} "
            f"({executor_name})"
        )

        def execute_indices(indices, executor_class):
            futures = {}
            with executor_class(max_workers=min(worker_count, len(indices))) as executor:
                for i in indices:
                    eval_dir = os.path.join(work_dir, f"eval_{i:04d}")
                    os.makedirs(eval_dir, exist_ok=True)
                    fut = executor.submit(
                        self._run_single, param_batch[i], eval_dir, save_traj
                    )
                    futures[fut] = i

                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        results[i] = EvalResult(
                            params=param_batch[i], properties={},
                            objective=LARGE_PENALTY, success=False,
                            error_msg=f"executor exception: {e}",
                        )

        if os.name == "nt":
            # Threads only coordinate external mpiexec/LAMMPS subprocesses, so
            # the GIL is not a bottleneck. Launching from the main process also
            # avoids Windows ProcessPool worker reuse breaking later MPI jobs.
            execute_indices(range(len(param_batch)), executor_cls)
        else:
            # Fresh Linux process waves keep each MPI evaluation isolated while
            # honoring the configured worker limit on a cluster node.
            for wave_start in range(0, len(param_batch), worker_count):
                wave_stop = min(wave_start + worker_count, len(param_batch))
                print(
                    f"  LAMMPS wave {wave_start // worker_count + 1}: "
                    f"candidates {wave_start + 1}-{wave_stop}"
                )
                execute_indices(range(wave_start, wave_stop), executor_cls)

        return results

    def feasibility_error(self, raw_params: Dict[str, float]) -> Optional[str]:
        """Return a cheap parameter/charge constraint error before LAMMPS."""
        try:
            resolved = self._resolve_params(raw_params)
        except Exception as exc:
            return f"param resolution failed: {exc}"
        return self._charge_feasible(resolved)

    def evaluate_replicates(self,
                            params: Dict[str, float],
                            work_dir: str,
                            seeds: List[int],
                            save_traj: bool = False,
                            reuse_existing: bool = False) -> List[EvalResult]:
        """
        Re-evaluate one parameter set with several independent bulk NPT seeds.

        This is intended for post-BO stability audits, not for the main BO loop.
        If the active recipe does not require bulk NPT (for example an
        adsorption-only objective), the seed override has no effect because the
        adsorption boxes are deterministic minimisations in the current workflow.
        """
        os.makedirs(work_dir, exist_ok=True)
        out: List[EvalResult] = []
        for seed in seeds:
            eval_dir = os.path.join(work_dir, f"seed_{int(seed)}")
            os.makedirs(eval_dir, exist_ok=True)
            if reuse_existing and not save_traj:
                recovered = self._recover_bulk_sublimation_result(
                    params, eval_dir
                )
                if recovered is not None:
                    out.append(recovered)
                    continue
            out.append(self._run_single(params, eval_dir, save_traj,
                                        seed_override=int(seed)))
        return out

    def _recover_bulk_sublimation_result(
            self,
            raw_params: Dict[str, float],
            eval_dir: str) -> Optional[EvalResult]:
        """Rebuild a completed bulk+sublimation evaluation from saved logs."""
        if not self.bulk_required or self.ads_enabled or self.compute_surface:
            return None

        try:
            resolved = self._resolve_params(raw_params)
        except Exception:
            return None
        if self._charge_feasible(resolved):
            return None

        bulk_log = os.path.join(eval_dir, "bulk", "lammps_stdout.log")
        if not os.path.isfile(bulk_log):
            return None
        try:
            with open(bulk_log, "r") as handle:
                bulk_props = self._parse_data_line(
                    handle.read(), "DATA_BULK:", self.bulk_col_map
                )
        except OSError:
            return None
        if bulk_props is None or self._bulk_sanity_check(bulk_props):
            return None

        properties: Dict[str, float] = dict(bulk_props)
        if self.sub_enabled:
            sub_log = os.path.join(
                eval_dir, "sublimation", "sub_single", "lammps_stdout.log"
            )
            if not os.path.isfile(sub_log):
                return None
            try:
                with open(sub_log, "r") as handle:
                    sub_props = self._parse_data_line(
                        handle.read(), "DATA_SUB_SINGLE:", ["E"]
                    )
            except OSError:
                return None
            n_atoms = self._read_atom_count(self.bulk_data)
            if sub_props is None or n_atoms is None or n_atoms <= 0:
                return None
            n_molecules = float(n_atoms) / float(self.sub_molecule_atoms)
            bulk_per_molecule = float(properties["pe"]) / n_molecules
            esub_kcal = float(sub_props["E"]) - bulk_per_molecule
            properties.update({
                "esub_proxy": esub_kcal * 4.184,
                "esub_proxy_kcal_mol": esub_kcal,
                "esub_single_pe": float(sub_props["E"]),
                "esub_bulk_pe_per_mol": bulk_per_molecule,
            })

        objective, per_prop = self._compute_objective(properties)
        if not np.isfinite(objective) or objective >= LARGE_PENALTY:
            return None
        obj_s, _ = self._compute_pareto_objectives(per_prop)
        return EvalResult(
            params=raw_params,
            properties=properties,
            objective=objective,
            success=True,
            per_property_error=per_prop,
            obj_structural=obj_s,
            obj_surface=float("nan"),
        )

    def evaluate_sublimation_proxy(self,
                                   params: Dict[str, float],
                                   work_dir: str,
                                   save_traj: bool = False) -> Optional[Dict[str, float]]:
        """
        Compute a 0 K sublimation/cohesive-energy proxy:

            E_sub_proxy = E_single_molecule - E_bulk / N_molecules

        Energies are LAMMPS potential energies after deterministic minimisation.
        The result is useful as a thermodynamic audit, but it is not yet a full
        finite-temperature experimental sublimation enthalpy.
        """
        if not self.sub_enabled:
            raise RuntimeError("sublimation.enabled is false in config")

        os.makedirs(work_dir, exist_ok=True)
        try:
            resolved = self._resolve_params(params)
        except Exception:
            return None
        if self._charge_feasible(resolved):
            return None

        return self._compute_sublimation_proxy_resolved(
            resolved, work_dir, save_traj)

    def _compute_sublimation_proxy_resolved(self,
                                            resolved: Dict[str, float],
                                            work_dir: str,
                                            save_traj: bool = False
                                            ) -> Optional[Dict[str, float]]:
        e_bulk = self._run_sublimation_box(
            resolved, work_dir, "sub_bulk", self.sub_bulk_data,
            self.sub_bulk_input, "DATA_SUB_BULK:", save_traj)
        if e_bulk is None:
            return None
        e_single = self._run_sublimation_box(
            resolved, work_dir, "sub_single", self.sub_single_data,
            self.sub_single_input, "DATA_SUB_SINGLE:", save_traj)
        if e_single is None:
            return None

        n_atoms = self._read_atom_count(self.sub_bulk_data)
        if n_atoms is None or n_atoms <= 0:
            return None
        n_molecules = float(n_atoms) / float(self.sub_molecule_atoms)
        e_bulk_per_mol = e_bulk / n_molecules
        esub_kcal_mol = e_single - e_bulk_per_mol
        esub_kj_mol = esub_kcal_mol * 4.184
        target_proxy = self.sub_target_kj - self.sub_thermal_correction_kj

        return {
            "E_bulk_kcal": e_bulk,
            "E_single_kcal": e_single,
            "n_molecules": n_molecules,
            "E_bulk_per_molecule_kcal": e_bulk_per_mol,
            "esub_proxy_kcal_mol": esub_kcal_mol,
            "esub_proxy_kj_mol": esub_kj_mol,
            "target_kj_mol": self.sub_target_kj,
            "thermal_correction_kj_mol": self.sub_thermal_correction_kj,
            "target_proxy_kj_mol": target_proxy,
            "error_vs_proxy_target_kj_mol": esub_kj_mol - target_proxy,
        }

    def _compute_sublimation_proxy_from_bulk_pe(
            self,
            resolved: Dict[str, float],
            work_dir: str,
            bulk_pe_kcal: float,
            save_traj: bool = False) -> Optional[Dict[str, float]]:
        """
        Compute the production-workflow sublimation proxy:

            E_sub_proxy = E_single_min - <PE_bulk_NPT> / N_molecules

        The bulk term is the same finite-temperature NPT average used for the
        structural properties, so seed-replicate audits measure the real noise
        that BO/NN/AL will see.
        """
        if bulk_pe_kcal is None or not np.isfinite(bulk_pe_kcal):
            return None

        e_single = self._run_sublimation_box(
            resolved, work_dir, "sub_single", self.sub_single_data,
            self.sub_single_input, "DATA_SUB_SINGLE:", save_traj)
        if e_single is None:
            return None

        n_atoms = self._read_atom_count(self.bulk_data)
        if n_atoms is None or n_atoms <= 0:
            return None
        n_molecules = float(n_atoms) / float(self.sub_molecule_atoms)
        e_bulk_per_mol = float(bulk_pe_kcal) / n_molecules
        esub_kcal_mol = e_single - e_bulk_per_mol
        esub_kj_mol = esub_kcal_mol * 4.184
        target_proxy = self.sub_target_kj - self.sub_thermal_correction_kj

        return {
            "bulk_energy_source": "npt_pe_mean",
            "E_bulk_kcal": float(bulk_pe_kcal),
            "E_single_kcal": e_single,
            "n_molecules": n_molecules,
            "E_bulk_per_molecule_kcal": e_bulk_per_mol,
            "esub_proxy_kcal_mol": esub_kcal_mol,
            "esub_proxy_kj_mol": esub_kj_mol,
            "target_kj_mol": self.sub_target_kj,
            "thermal_correction_kj_mol": self.sub_thermal_correction_kj,
            "target_proxy_kj_mol": target_proxy,
            "error_vs_proxy_target_kj_mol": esub_kj_mol - target_proxy,
        }

    # ====================================================================== #
    # Private: evaluation pipeline                                            #
    # ====================================================================== #

    def _run_bulk(self,
                  resolved: Dict[str, float],
                  eval_dir: str,
                  save_traj: bool = False,
                  seed_override: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Run the bulk NPT stage and return parsed DATA_BULK properties."""
        _sv = 1 if save_traj else 0
        bulk_dir = os.path.join(eval_dir, "bulk")
        os.makedirs(bulk_dir, exist_ok=True)
        shutil.copy2(self.bulk_input, bulk_dir)
        shutil.copy2(self.bulk_data, bulk_dir)
        self._build_pair_coeffs(resolved, bulk_dir)

        bulk_vars = {
            "bulk_data":      os.path.basename(self.bulk_data),
            "nx":             self.bulk_nx,
            "ny":             self.bulk_ny,
            "nz":             self.bulk_nz,
            "timestep_value": self.timestep,
            "cutoff":         self.cutoff,
            "npt_seed":       seed_override if seed_override is not None else self.bulk_npt_seed,
            "equil_steps":    self.bulk_equil,
            "prod_steps":     self.bulk_prod,
            "temperature":    self.bulk_temperature,
            "pressure":       self.bulk_pressure,
            "save_traj":      _sv,
        }
        if self.use_charge:
            bulk_vars["kspace_accuracy"] = self.kspace_accuracy

        return self._run_one(
            cwd=bulk_dir,
            input_file=os.path.basename(self.bulk_input),
            extra_vars=bulk_vars,
            data_marker="DATA_BULK:",
            output_map=self.bulk_col_map,
        )

    def _run_single(self,
                    raw_params: Dict[str, float],
                    eval_dir: str,
                    save_traj: bool = False,
                    seed_override: Optional[int] = None) -> EvalResult:
        """
        Run one evaluation: resolve params, then execute only required property
        stages (bulk, adsorption, sublimation, surface).

        raw_params : flat dict from BO sampler (free parameters only).
        eval_dir   : dedicated working directory for this evaluation.
        save_traj  : passed through to LAMMPS save_traj variable.
        """
        # ------------------------------------------------------------------ #
        # Step 0: Resolve all parameters                                      #
        # Applies derived_params expressions and charge neutrality.           #
        # ------------------------------------------------------------------ #
        try:
            resolved = self._resolve_params(raw_params)
        except Exception as e:
            return EvalResult(
                params=raw_params, properties={},
                objective=LARGE_PENALTY, success=False,
                error_msg=f"param resolution failed: {e}",
            )

        # Hard charge feasibility gate (q>0 for H, |q|<abs_max for all types).
        # Cheap rejection BEFORE LAMMPS keeps BO inside the allowed charge box.
        cerr = self._charge_feasible(resolved)
        if cerr:
            return EvalResult(
                params=raw_params, properties={},
                objective=LARGE_PENALTY, success=False,
                error_msg=cerr,
            )

        stage_result = self.property_plan.execute(
            self,
            PropertyEvaluationContext(
                resolved=resolved,
                eval_dir=eval_dir,
                save_traj=save_traj,
                seed_override=seed_override,
            ),
        )
        properties = stage_result.properties
        if not stage_result.success:
            return EvalResult(
                params=raw_params,
                properties=properties,
                objective=LARGE_PENALTY,
                success=False,
                error_msg=stage_result.error,
            )

        objective, per_prop = self._compute_objective(properties)
        obj_s, obj_surf = self._compute_pareto_objectives(per_prop)
        return EvalResult(
            params=raw_params, properties=properties,
            objective=objective, success=True,
            per_property_error=per_prop,
            obj_structural=obj_s, obj_surface=obj_surf,
        )

    def _run_surf(self,
                 resolved:   Dict[str, float],
                 eval_dir:   str,
                 tag:        str,
                 slab_path:  str,
                 extra_base: Dict) -> Optional[Dict]:
        """
  2. Surf NVT 300 K (complete + split run in parallel) -> surf_energy
        Called concurrently for both tags inside ThreadPoolExecutor.
        """
        sdir = os.path.join(eval_dir, f"surf_{tag}")
        os.makedirs(sdir, exist_ok=True)
        shutil.copy2(self.surf_input, sdir)
        shutil.copy2(slab_path, os.path.join(sdir, "slab.data"))
        self._build_pair_coeffs(resolved, sdir)
        return self._run_one(
            cwd=sdir,
            input_file=os.path.basename(self.surf_input),
            extra_vars={"slab_file": "slab.data", "slab_tag": tag, **extra_base},
            data_marker="DATA_SURF:",
            output_map=None,
        )

    # ====================================================================== #
    # Private: parameter resolution                                           #
    # ====================================================================== #

    def _resolve_params(self, raw_params: Dict[str, float]) -> Dict[str, float]:
        """
        Resolve all parameters from raw BO samples to a complete parameter set.

        Steps
        -----
        1. Apply derived_params algebraic expressions (e.g. Fe_body_epsilon = Fe_corner_epsilon).
        2. Apply charge neutrality constraint (if charge.enabled: true).

        Parameters
        ----------
        raw_params : dict
            Free BO parameters from the sampler.
            Keys: "{label}_{param}", e.g. "Fe_corner_epsilon", "Fe_body_sigma".

        Returns
        -------
        dict
            Same keys as raw_params, plus any derived/charge-constrained keys.
        """
        resolved = dict(raw_params)

        # Step 0: inline fixed/formula params declared in atom_types.
        #   scalar value  -> fixed constant (e.g. sigma: 3.2963)
        #   string        -> formula referencing {label}_{param}
        #                    (e.g. epsilon: "bhN2_epsilon - 1.0", charge: "-bhA_charge")
        _formulas = []
        for at in self.atom_types:
            label = at["label"]
            for pname, spec in at["params"].items():
                key = f"{label}_{pname}"
                if key in resolved:
                    continue                      # free param, already sampled
                if isinstance(spec, bool):
                    continue
                if isinstance(spec, (int, float)):
                    resolved[key] = float(spec)   # fixed constant
                elif isinstance(spec, str):
                    _formulas.append((key, spec))  # formula -> resolve below
        # Resolve formulas iteratively (handles formula-references-formula chains)
        for _ in range(len(_formulas) + 1):
            for key, expr in _formulas:
                if key in resolved:
                    continue
                try:
                    resolved[key] = float(eval(expr, {"__builtins__": {}}, resolved))
                except Exception:
                    pass
            if all(k in resolved for k, _ in _formulas):
                break
        _missing = [k for k, _ in _formulas if k not in resolved]
        if _missing:
            raise ValueError(f"inline formula params unresolved (check references): {_missing}")

        # Step 1: derived_params expressions
        for constraint in self.derived_params_cfg:
            target_key = constraint["target"]
            expr       = constraint["expression"]
            comment    = constraint.get("comment", "")
            try:
                # eval in a restricted namespace: only resolved values available
                resolved[target_key] = float(eval(expr, {"__builtins__": {}}, resolved))
            except Exception as e:
                raise ValueError(
                    f"derived_params: failed to evaluate '{target_key}' = '{expr}'"
                    f"{(' (' + comment + ')') if comment else ''}. Error: {e}"
                )

        # Step 2: charge neutrality
        if self.use_charge:
            resolved = self._apply_charge_neutrality(resolved)

        return resolved

    def _apply_charge_neutrality(self,
                                 params: Dict[str, float]) -> Dict[str, float]:
        """
        Derive the charge of one atom type from the charge neutrality condition:
            q_derived * n_derived = -(sum_i q_i * n_i) for all other types i

        The type to derive is specified by charge.neutrality_constraint.derive_from_type
        in config.yaml.

        Parameters
        ----------
        params : dict
            Resolved params dict; must contain "{label}_charge" for all types
            except the derived type.

        Returns
        -------
        dict
            Same dict with the derived charge key added/overwritten.
        """
        result  = dict(params)
        n_cfg   = self.charge_cfg["neutrality_constraint"]

        if not n_cfg["enabled"]:
            return result

        derive_idx   = n_cfg["derive_from_type"]          # 1-indexed
        derive_type  = self.atom_types[derive_idx - 1]
        derive_label = derive_type["label"]
        derive_key   = f"{derive_label}_charge"

        n_derive = self.bulk_type_counts.get(derive_idx, 0)
        if n_derive == 0:
            raise ValueError(
                f"charge neutrality: type {derive_idx} ({derive_label}) has "
                f"0 atoms in bulk data file; cannot derive charge. "
                f"Check derive_from_type in config."
            )

        # Sum contributions from all other types
        charge_sum = 0.0
        for at in self.atom_types:
            t_idx = at["type"]
            if t_idx == derive_idx:
                continue
            label = at["label"]
            q_key = f"{label}_charge"
            if q_key not in result:
                raise KeyError(
                    f"charge neutrality: expected free parameter '{q_key}' in params "
                    f"but it was not found. Ensure atom_type '{label}' has charge "
                    f"bounds defined in config.yaml and is not the derived type."
                )
            n_i = self.bulk_type_counts.get(t_idx, 0)
            charge_sum += result[q_key] * n_i

        result[derive_key] = -charge_sum / n_derive
        return result

    def _charge_feasible(self, resolved: Dict[str, float]) -> Optional[str]:
        """
        Hard charge constraints from config.charge.constraints, checked BEFORE
        running LAMMPS (cheap rejection of infeasible BO points).

        constraints:
          abs_max         : |q| must be < this for EVERY atom type
          positive_labels : these atom types must have q > 0

        These cannot be expressed as simple BO box bounds because the
        neutrality-derived charge (e.g. bhHn) and the +offset-derived ring-H
        charges are computed, not sampled. Returns an error string if any
        constraint is violated, else None.
        """
        if not self.use_charge:
            return None
        cons = self.charge_cfg.get("constraints")
        if not cons:
            return None
        abs_max    = cons.get("abs_max")
        pos_labels = set(cons.get("positive_labels", []))
        for at in self.atom_types:
            label = at["label"]
            q = resolved.get(f"{label}_charge")
            if q is None:
                continue
            if abs_max is not None and abs(q) > float(abs_max):
                return (f"charge infeasible: |q({label})|={abs(q):.3f} "
                        f"> abs_max {abs_max}")
            if label in pos_labels and q <= 0.0:
                return f"charge infeasible: q({label})={q:.3f} must be > 0"
        return None

    # ====================================================================== #
    # Private: LAMMPS force field file generation                            #
    # ====================================================================== #

    def _build_pair_coeffs(self,
                           resolved: Dict[str, float],
                           cwd: str) -> str:
        """
        Write pair_coeffs.lmp into cwd with all pair_coeff and charge commands.

        The LAMMPS input script must contain:
            include pair_coeffs.lmp

        File contents (in order):
          1. pair_modify mix {rule}        (if mixing_rule != none)
          2. pair_coeff i i epsilon sigma  (same-type, for every atom type)
          3. pair_coeff i j epsilon sigma  (explicit cross pairs, mixing_rule: none only)
          4. set type N charge Q           (if charge.enabled: true, for every type)

        Parameters
        ----------
        resolved : dict
            Fully resolved parameter dict from _resolve_params().
        cwd : str
            Directory where pair_coeffs.lmp will be written.

        Returns
        -------
        str : absolute path to the written file.
        """
        lines = [
            "# pair_coeffs.lmp -- auto-generated by lammps_interface.py v8",
            "# DO NOT EDIT: overwritten before each LAMMPS call",
            "",
        ]

        # 1. Mixing rule (must precede pair_coeff for LAMMPS to apply it)
        if self.mixing_rule != "none":
            lines.append(f"pair_modify mix {self.mixing_rule}")
            lines.append("")

        # 2. Same-type pair_coeff i i epsilon sigma
        for at in self.atom_types:
            t     = at["type"]
            label = at["label"]
            eps_key = f"{label}_epsilon"
            sig_key = f"{label}_sigma"
            if eps_key not in resolved or sig_key not in resolved:
                raise KeyError(
                    f"pair_coeffs: expected '{eps_key}' and '{sig_key}' in resolved "
                    f"params for atom type {t} ({label}). "
                    f"Check atom_types param bounds in config."
                )
            eps = resolved[eps_key]
            sig = resolved[sig_key]
            lines.append(f"pair_coeff {t} {t} {eps:.10f} {sig:.10f}")

        lines.append("")

        # 3. Explicit cross-pair pair_coeff (only when mixing_rule: none)
        if self.mixing_rule == "none" and self.explicit_pairs_cfg:
            lines.append("# Explicit cross-type pair_coeff (mixing_rule: none)")
            for ep_cfg in self.explicit_pairs_cfg:
                t1, t2   = ep_cfg["types"]
                eps_key  = f"cross_{t1}_{t2}_epsilon"
                sig_key  = f"cross_{t1}_{t2}_sigma"
                if eps_key not in resolved or sig_key not in resolved:
                    raise KeyError(
                        f"pair_coeffs: explicit pair ({t1},{t2}) requires "
                        f"'{eps_key}' and '{sig_key}' in resolved params. "
                        f"Check explicit_pairs bounds in config."
                    )
                eps = resolved[eps_key]
                sig = resolved[sig_key]
                lines.append(f"pair_coeff {t1} {t2} {eps:.10f} {sig:.10f}")
            lines.append("")

        # 4. Set charges (charged systems only, after pair_coeff)
        if self.use_charge:
            lines.append("# Assign per-type charges")
            for at in self.atom_types:
                t     = at["type"]
                label = at["label"]
                q_key = f"{label}_charge"
                if q_key not in resolved:
                    raise KeyError(
                        f"pair_coeffs: charge enabled but '{q_key}' missing from "
                        f"resolved params for type {t} ({label})."
                    )
                q = resolved[q_key]
                lines.append(f"set type {t} charge {q:.10f}")
            lines.append("")

        out_path = os.path.join(cwd, "pair_coeffs.lmp")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return out_path

    # ====================================================================== #
    # Private: adsorption-energy (3-box) helpers                              #
    # ====================================================================== #

    @staticmethod
    def _read_type_labels(data_path: str) -> Dict[int, str]:
        """
        Parse the Masses section of a data file -> {type_index: label}, taking
        the label from the trailing "# label" comment (msi2lmp / CGCMM format).
        Used to map BTAH labels onto each adsorption system's own type order.
        """
        labels: Dict[int, str] = {}
        in_masses = False
        try:
            with open(data_path) as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("Masses"):
                        in_masses = True
                        continue
                    if not in_masses:
                        continue
                    if not s:
                        continue
                    parts = s.split()
                    if not parts[0].lstrip("-").isdigit():
                        break  # next section header ends Masses
                    idx = int(parts[0])
                    lab = s.split("#", 1)[1].strip().split()[0] if "#" in s else None
                    labels[idx] = lab
        except OSError:
            pass
        return labels

    def _write_pair_coeffs_system(self,
                                  resolved: Dict[str, float],
                                  cwd: str,
                                  type_labels: Dict[int, str],
                                  has_charge: bool) -> str:
        """
        Write pair_coeffs.lmp for one adsorption system, mapping BTAH labels onto
        that system's type indices. The metal type (self.metal_label) keeps its
        data-file LJ (no pair_coeff) and is set to charge 0.
        """
        lines = ["# pair_coeffs.lmp (adsorption system) -- auto-generated", ""]
        if self.mixing_rule != "none":
            lines.append(f"pair_modify mix {self.mixing_rule}")
            lines.append("")
        for t in sorted(type_labels):
            label = type_labels[t]
            if label == self.metal_label:
                continue  # metal LJ kept from data file
            eps = resolved.get(f"{label}_epsilon")
            sig = resolved.get(f"{label}_sigma")
            if eps is None or sig is None:
                raise KeyError(f"adsorption pair_coeffs: missing eps/sig for "
                               f"type {t} (label '{label}')")
            lines.append(f"pair_coeff {t} {t} {eps:.10f} {sig:.10f}")
        lines.append("")
        if has_charge:
            for t in sorted(type_labels):
                label = type_labels[t]
                if label == self.metal_label:
                    lines.append(f"set type {t} charge 0.0")   # metal uncharged
                else:
                    q = resolved.get(f"{label}_charge")
                    if q is None:
                        raise KeyError(f"adsorption charge: missing {label}_charge")
                    lines.append(f"set type {t} charge {q:.10f}")
            lines.append("")
        out = os.path.join(cwd, "pair_coeffs.lmp")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        return out

    def _run_ads_box(self, resolved, parent_dir, name, data_path, input_path,
                     has_charge: bool, save_traj: bool) -> Optional[float]:
        """Run one adsorption box (NVT), return time-averaged PE [kcal/mol] or None."""
        d = os.path.join(parent_dir, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(input_path, d)
        shutil.copy2(data_path, d)
        type_labels = self._read_type_labels(data_path)
        self._write_pair_coeffs_system(resolved, d, type_labels, has_charge)
        extra_vars = {
            "data":           os.path.basename(data_path),
            "cutoff":         self.ads_cutoff,
            "timestep_value": self.ads_timestep,
            "temp":           self.ads_temp,
            "seed":           self.ads_seed,
            "equil_steps":    self.ads_equil,
            "prod_steps":     self.ads_prod,
            "save_traj":      1 if save_traj else 0,
        }
        if has_charge:
            extra_vars["kspace_accuracy"] = self.ads_kspace
        props = self._run_one(cwd=d, input_file=os.path.basename(input_path),
                              extra_vars=extra_vars, data_marker="DATA_AD:",
                              output_map=["E"])
        return None if props is None else props["E"]

    def _run_adsorption(self, resolved, eval_dir, save_traj) -> Optional[Dict[str, float]]:
        """Return adsorption energy and its three energy components."""
        e_complex = self._run_ads_box(resolved, eval_dir, "ad_complex",
                                      self.ads_complex, self.ads_complex_input,
                                      has_charge=True, save_traj=save_traj)
        if e_complex is None:
            return None
        e_mol = self._run_ads_box(resolved, eval_dir, "ad_mol",
                                  self.ads_mol, self.ads_mol_input,
                                  has_charge=True, save_traj=save_traj)
        if e_mol is None:
            return None
        # Re-run the parameter-independent slab when trajectories are requested;
        # the cached reference calculation intentionally does not save files.
        e_slab = None if save_traj else self._slab_pe
        if e_slab is None:
            e_slab = self._run_ads_box(resolved, eval_dir, "ad_slab",
                                       self.ads_slab, self.ads_slab_input,
                                       has_charge=False, save_traj=save_traj)
            if e_slab is None:
                return None
        return {
            "ead": e_complex - e_slab - e_mol,
            "ead_complex_pe": e_complex,
            "ead_slab_pe": e_slab,
            "ead_mol_pe": e_mol,
        }

    def _run_sublimation_box(self, resolved, parent_dir, name, data_path,
                             input_path, marker, save_traj) -> Optional[float]:
        """Run one deterministic sublimation proxy box, return PE [kcal/mol]."""
        d = os.path.join(parent_dir, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(input_path, d)
        shutil.copy2(data_path, d)
        self._build_pair_coeffs(resolved, d)
        extra_vars = {
            "data": os.path.basename(data_path),
            "cutoff": self.sub_cutoff,
            "kspace_accuracy": self.sub_kspace,
            "save_traj": 1 if save_traj else 0,
        }
        props = self._run_one(cwd=d, input_file=os.path.basename(input_path),
                              extra_vars=extra_vars, data_marker=marker,
                              output_map=["E"])
        return None if props is None else props["E"]

    # ====================================================================== #
    # Private: LAMMPS execution                                              #
    # ====================================================================== #

    def _scheduler_node_index(self, cwd: str) -> int:
        identifiers = re.findall(r"(?:eval|candidate)_(\d+)", str(cwd))
        if identifiers:
            index = int(identifiers[-1])
        else:
            index = zlib.crc32(str(cwd).encode("utf-8"))
        return index % self.scheduler_node_count

    def _mpi_prefix(self, ranks: int, cwd: str = "") -> List[str]:
        """Build a local or scheduler-aware MPI launcher prefix."""
        if self.scheduler_launcher:
            cpus = int(ranks) * int(self.omp_threads)
            prefix = [
                str(self.scheduler_launcher),
                "--overlap",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task", str(cpus),
            ]
            if self.scheduler_node_count > 1:
                prefix.append(f"--relative={self._scheduler_node_index(cwd)}")
            return prefix + [
                sys.executable,
                "-m", "workflow.mpi_local_exec",
                "--launcher", self.mpiexec,
                "--ranks", str(ranks),
                "--slots", str(self.workers_per_node),
                "--",
            ]
        launcher = os.path.basename(str(self.mpiexec)).lower()
        if launcher in {"srun", "srun.exe"}:
            return [
                self.mpiexec,
                "--exact",
                "--exclusive",
                "--nodes=1",
                "--ntasks", str(ranks),
                "--cpus-per-task", str(self.omp_threads),
            ]
        return [self.mpiexec, "-n", str(ranks)]

    def _run_one(self,
                 cwd: str,
                 input_file: str,
                 extra_vars: Dict,
                 data_marker: str,
                 output_map: Optional[List[str]]) -> Optional[Dict]:
        """
        Execute one LAMMPS call and parse the DATA_* output line.

        Parameters
        ----------
        cwd         : working directory (input script + data file already copied here)
        input_file  : LAMMPS script filename (basename only)
        extra_vars  : dict of LAMMPS -var key value pairs (infrastructure vars only;
                      pair params are in pair_coeffs.lmp)
        data_marker : prefix of the LAMMPS print line to parse, e.g. "DATA_BULK:"
        output_map  : if list of str -> positional float columns
                      if None        -> key=value token parsing (DATA_SURF format)

        Returns
        -------
        dict or None (None = LAMMPS failed or marker not found in stdout)
        """
        # Build -var arguments
        var_args = []
        for k, v in extra_vars.items():
            var_args.extend(["-var", str(k), str(v)])

        # Build MPI command
        if self.use_mpi:
            cmd = self._mpi_prefix(self.cores, cwd) + [
                self.lammps_exe, "-in", input_file,
            ] + var_args
        else:
            # use_mpi=False (--no-mpi): single-rank MPI via mpiexec -n 1.
            # Requires Intel MPI module: module load mpi/2021.15
            # I_MPI_FABRICS=shm forces shared-memory transport; bypasses OFI/InfiniBand
            # which may be unavailable on login/management nodes.
            cmd = self._mpi_prefix(1, cwd) + [
                self.lammps_exe, "-in", input_file,
            ] + var_args

        _env = {**os.environ,
                "OMP_NUM_THREADS": str(self.omp_threads),
                "MKL_NUM_THREADS": str(self.omp_threads)}
        if not self.use_mpi and "I_MPI_FABRICS" not in os.environ:
            _env["I_MPI_FABRICS"] = "shm"

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=cwd, env=_env,
            )
        except subprocess.TimeoutExpired:
            self._dump_diagnostic(cwd, f"TIMEOUT after {self.timeout}s")
            return None
        except Exception as e:
            self._dump_diagnostic(cwd, f"EXCEPTION launching LAMMPS: {e}")
            return None

        # Always save logs (essential for debugging failed BO points)
        with open(os.path.join(cwd, "lammps_stdout.log"), "w") as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(cwd, "lammps_stderr.log"), "w") as f:
                f.write(result.stderr)

        if result.returncode != 0:
            return None

        return self._parse_data_line(result.stdout, data_marker, output_map)

    # ====================================================================== #
    # Private: output parsing                                                 #
    # ====================================================================== #

    @staticmethod
    def _parse_data_line(stdout: str,
                         marker: str,
                         output_map: Optional[List[str]]) -> Optional[Dict]:
        """
        Find the DATA_* line in LAMMPS stdout and parse it.

        Two modes
        ---------
        output_map is a list of str:
            Positional float columns.  e.g. DATA_BULK: 2.83 2.87 ... 7.87
            Returns {col_name: float_value} using output_map as column names.

        output_map is None:
            Key=value token format.  e.g. DATA_SURF: tag=complete E_slab=-12345.6 A_xy=52.3
            Returns {key: value} parsed from "key=value" tokens.
            Numeric values are converted to float; non-numeric kept as str.
        """
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith(marker):
                continue
            body   = line[len(marker):].strip()
            tokens = body.split()

            if output_map is not None:
                # Positional float parsing
                try:
                    values = [float(t) for t in tokens]
                except ValueError:
                    return None
                if len(values) != len(output_map):
                    return None
                return dict(zip(output_map, values))
            else:
                # key=value token parsing
                result = {}
                for tok in tokens:
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v
                return result

        return None   # marker not found

    # ====================================================================== #
    # Private: objective computation                                          #
    # ====================================================================== #

    def _compute_objective(self,
                           properties: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute the weighted objective across all active targets.

        Formula (true weighted RMSE of relative errors):
            objective = sqrt(sum_i w_i * rel_err_i^2 / sum_i w_i)

        Dividing by weight sum makes the objective dimensionless and comparable
        across systems with different numbers of active targets or different
        weight configurations.  The square/sqrt gives extra penalty to any single
        property that deviates severely (RMSE vs MAE behaviour).

        # compute_surface: false -> return the currently requested properties

        Returns
        -------
        (objective_scalar, per_property_error_percent_dict)
        """
        sq_sum     = 0.0
        weight_sum = 0.0
        per_prop   = {}

        for prop, info in self.targets.items():
            if prop == "surf_energy" and not self.compute_surface:
                continue
            if prop == "ead" and not self.ads_enabled:
                continue
            if prop == "esub_proxy" and not self.sub_enabled:
                continue

            target = float(info["value"])
            weight = float(info.get("weight", 1.0))

            if weight == 0.0:
                continue

            if prop not in properties:
                return LARGE_PENALTY, {}

            calc = properties[prop]
            if not np.isfinite(calc):
                return LARGE_PENALTY, {}

            rel_err = abs(calc - target) / (abs(target) + 1e-10)
            per_prop[prop] = rel_err * 100.0
            sq_sum     += weight * rel_err ** 2
            weight_sum += weight

        if weight_sum < 1e-12:
            return LARGE_PENALTY, {}

        objective = float(np.sqrt(sq_sum / weight_sum))
        return objective, per_prop

    def _compute_pareto_objectives(
            self,
            per_prop: Dict[str, float]) -> Tuple[float, float]:
        """
        Compute the two Pareto group objectives from per-property errors.

        Group 1 (obj_structural):
            sqrt(sum_i w_i * (err_i/100)^2 / sum_i w_i) for all non-surface targets
            Same formula as _compute_objective: true weighted RMSE of relative errors.
            Normalising by sum_i w_i makes it independent of the number of targets,
            so different crystal systems can be compared on the same scale.

        Group 2 (obj_surface):
            surf_energy relative error [%] (nan when compute_surface=false).

        Stored in EvalResult and used by optimizer.py for qNEHVI + post-hoc Pareto.
        """
        sq_sum        = 0.0
        struct_weight = 0.0

        for prop, info in self.targets.items():
            if prop == "surf_energy":
                continue
            weight = float(info.get("weight", 1.0))
            if weight == 0.0 or prop not in per_prop:
                continue
            rel_err        = per_prop[prop] / 100.0   # convert % back to fraction
            sq_sum        += weight * rel_err ** 2
            struct_weight += weight

        obj_structural = (float(np.sqrt(sq_sum / struct_weight))
                          if struct_weight > 1e-12 else float("nan"))
        obj_surface    = per_prop.get("surf_energy", float("nan"))

        return obj_structural, obj_surface

    # ====================================================================== #
    # Private: sanity checks                                                  #
    # ====================================================================== #

    def _bulk_sanity_check(self, bulk_props: Dict[str, float]) -> Optional[str]:
        """
        Reject bulk results where NPT has gone clearly runaway.

        Gates
        -----
        Lattice parameters a, b, c : each must lie within [0.5x, 2.0x] of its
                                     OWN target (ref_lat[a/b/c]). This supports
                                     OWN target. This supports anisotropic cells such as molecular
                                     crystals, where a single ref would wrongly
                                     flag the long axes.
        Density                     : must lie within [0.2x, 3.0x] of ref_density.

        Returns
        -------
        str  : error message if any gate fires.
        None : all gates passed.
        """
        for lp in ("a", "b", "c"):
            ref = self.ref_lat.get(lp)
            if ref is None:
                continue
            lo, hi = 0.5 * ref, 2.0 * ref
            val = bulk_props.get(lp)
            if val is None or not np.isfinite(val):
                return f"bulk sanity: '{lp}' is non-finite or missing"
            if not (lo < val < hi):
                return (f"bulk sanity: {lp}={val:.4f} A outside physical range "
                        f"[{lo:.3f}, {hi:.3f}] A; NPT likely runaway")

        if self.ref_density is not None:
            rho = bulk_props.get("density")
            if rho is None or not np.isfinite(rho):
                return "bulk sanity: 'density' is non-finite or missing"
            lo_d, hi_d = 0.2 * self.ref_density, 3.0 * self.ref_density
            if not (lo_d < rho < hi_d):
                return (f"bulk sanity: density={rho:.4f} g/cm3 outside physical range "
                        f"[{lo_d:.3f}, {hi_d:.3f}] g/cm3")

        return None

    # ====================================================================== #
    # Private: static utilities                                               #
    # ====================================================================== #

    @staticmethod
    def _read_type_counts(data_path: str) -> Dict[int, int]:
        """
        Parse a LAMMPS data file and return {atom_type_index: count} dict.

        Reads the Atoms section. Assumes atom_style full:
            atom-id  mol-id  atom-type  charge  x  y  z
        atom-type is the 3rd column (0-indexed: parts[2]).

        Returns empty dict on any read failure (non-fatal; charge neutrality
        will raise a descriptive error if counts are needed but missing).
        """
        counts: Dict[int, int] = {}
        in_atoms = False

        try:
            with open(data_path) as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue

                    # Detect Atoms section header
                    if s == "Atoms" or s.startswith("Atoms "):
                        in_atoms = True
                        continue

                    if in_atoms:
                        # Another section header (non-numeric first char) ends Atoms
                        if s and not s[0].isdigit():
                            break
                        parts = s.split()
                        if len(parts) >= 3:
                            try:
                                atom_type = int(parts[2])   # col 2: atom-type in full style
                                counts[atom_type] = counts.get(atom_type, 0) + 1
                            except (ValueError, IndexError):
                                pass
        except OSError:
            pass

        return counts

    @staticmethod
    def _read_atom_count(data_path: str) -> Optional[int]:
        """Return total atom count from LAMMPS data file header."""
        try:
            with open(data_path) as f:
                for line in f:
                    s = line.strip()
                    if s.endswith("atoms") and not s.startswith("#"):
                        try:
                            return int(s.split()[0])
                        except (ValueError, IndexError):
                            continue
        except OSError:
            pass
        return None

    @staticmethod
    def _dump_diagnostic(cwd: str, msg: str) -> None:
        """Write a diagnostic.log to cwd for post-mortem debugging."""
        try:
            with open(os.path.join(cwd, "diagnostic.log"), "w") as f:
                f.write(msg + "\n")
        except OSError:
            pass
