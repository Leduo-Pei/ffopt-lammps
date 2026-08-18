"""Typed runtime defaults used by the public ``ffopt.in`` interface.

New projects are compiled from these defaults. Keeping defaults in Python gives one canonical
source and prevents local/cluster/scientific settings from being duplicated
across a directory tree of user-edited files.
"""

from __future__ import annotations

import copy
from typing import Any


LOCAL_MACHINE_DEFAULTS: dict[str, Any] = {
    "machine": {"backend": "local"},
    "lammps": {
        "executable": "lmp",
        "mpiexec": "mpiexec",
        "timeout": 21600,
    },
    "parallel": {
        "max_workers": 1,
        "cores_per_worker": 1,
        "omp_threads_per_worker": 1,
        "use_mpi": False,
    },
    "machine_learning": {"device": "auto"},
}


SLURM_MACHINE_DEFAULTS: dict[str, Any] = {
    "machine": {"backend": "slurm"},
    "lammps": {
        "executable": "lmp",
        "mpiexec": "srun",
        "timeout": 21600,
    },
    "parallel": {
        "max_workers": 1,
        "cores_per_worker": 1,
        "omp_threads_per_worker": 1,
        "use_mpi": True,
    },
    "machine_learning": {"device": "auto"},
    "cluster": {
        "env_setup": [],
        "bo": {
            "partition": "", "qos": "", "nodes": 1, "cores": 1,
            "time": "24:00:00", "mem": "0",
        },
        "nn": {
            "partition": "", "qos": "", "nodes": 1, "cores": 1,
            "time": "24:00:00", "mem": "0", "gpu": 0,
        },
        "al": {
            "partition": "", "qos": "", "nodes": 1, "cores": 1,
            "time": "24:00:00", "mem": "0", "gpu": 0,
        },
    },
}


METHOD_DEFAULTS: dict[str, Any] = {
    "optimization": {
        "seed_params": "",
        "method": "auto",
        # Scientific BO batch size.  This must not depend on the number of
        # workers available on a particular machine.
        "batch_size": 48,
        "accuracy_priority": False,
        "n_initial": 48,
        "n_bo_iterations": 200,
        "objective": "weighted_rmse",
        "pareto": {
            "mode": "posthoc",
            "reference_point": {"structural": 30.0, "surface": 50.0},
            "knee_method": "min_distance",
        },
        "turbo": {
            "n_trust_regions": 1,
            "length_init": 0.8,
            "length_min": 0.005,
            "length_max": 1.6,
        },
        "saasbo": {"num_samples": 256, "warmup_steps": 512},
        "early_stop": {
            "enabled": True,
            "patience": 30,
            "min_improvement": 0.001,
        },
        "exploration": {
            "every_n_rounds": 5,
            "n_random": 16,
            "n_uncertain": 2,
        },
        "feasibility": {
            "penalty_cap_multiplier": 3.0,
            "classifier": True,
        },
        "random_seed": 42,
        "progress_report": {
            "enabled": True,
            "property_table_every": 1,
            "save_csv": True,
        },
        "stability_audit": {
            "enabled": True,
            "top_k": 20,
            "seeds": [101, 202, 303],
            "max_objective_std": 0.05,
            "max_property_rel_std": 0.05,
            "noise_penalty": 1.0,
            "failure_penalty": 10.0,
        },
    },
    "checkpoint": {"enabled": True, "interval": 10, "directory": "checkpoints"},
    "nn": {
        "enabled": True,
        "model": "mlp_ensemble",
        "property_models": {},
        "physics_features": False,
        "derived_charge_feature": True,
        "ensemble_size": 8,
        "hidden_layers": [256, 128, 128, 64],
        "activation": "silu",
        "train_top_fraction": 1.0,
        "prefer_stable_results": True,
        "training_data": {
            "mode": "core_buffer",
            "core_objective_max": 1.0,
            "buffer_objective_max": 2.0,
            "core_weight": 3,
            "additional_core_files": [],
        },
        "exclude_properties": [],
        "learning_rate": 5.0e-4,
        "loss": "huber",
        "weight_decay": 1.0e-4,
        "batch_size": 128,
        "max_epochs": 1200,
        "early_stopping_patience": 150,
        "val_fraction": 0.15,
        "test_fraction": 0.10,
        "optimize": {"n_restarts": 10, "method": "L-BFGS-B"},
    },
    "active_learning": {
        "enabled": True,
        "n_rounds": 2,
        "n_candidates_per_round": 20,
        "uncertainty_threshold": 0.03,
        "candidate_sampling": "sobol",
        "sampling_domain": "core_envelope",
        "sampling_objective_max": 1.0,
        "envelope_margin": 0.15,
        "n_candidate_pool": 16384,
        "local_sampling_fraction": 0.75,
        "local_elite_objective_max": 1.0,
        "local_radii": [0.01, 0.025, 0.05],
        "exploitation_fraction": 0.80,
        "uncertainty_penalty": 0.50,
        "promising_quantile": 0.20,
        "preserve_focused_hybrid": True,
        "direct_objective_weight": 0.65,
        "direct_objective_max": 2.0,
        "objective_bias": 0.25,
    },
}


def machine_defaults(name: str) -> dict[str, Any]:
    """Return isolated defaults for a conventional machine profile name."""
    if name == "local":
        return copy.deepcopy(LOCAL_MACHINE_DEFAULTS)
    if name == "cluster":
        return copy.deepcopy(SLURM_MACHINE_DEFAULTS)
    raise ValueError(f"No built-in machine profile named {name!r}")


def method_defaults() -> dict[str, Any]:
    """Return an isolated copy of the BO/ANN/AL defaults."""
    return copy.deepcopy(METHOD_DEFAULTS)
