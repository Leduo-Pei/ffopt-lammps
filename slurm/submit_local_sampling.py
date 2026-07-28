#!/usr/bin/env python3
"""Submit or resume a modular multi-seed BTAH sampling campaign."""

from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--elite-centers", type=int, default=None)
    parser.add_argument("--center-selection", choices=["top", "diverse"], default=None)
    parser.add_argument("--center-pool-multiplier", type=int, default=None)
    parser.add_argument("--radii", type=float, nargs="+", default=None)
    parser.add_argument("--global-fraction", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--design-seed", type=int, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--dependency", default=None,
        help="Optional SLURM dependency, for example afterany:69015."
    )
    parser.add_argument(
        "--keep-work", action="store_true",
        help="Retain per-seed LAMMPS work directories for staged seed expansion."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    sampling = dict(config.get("sampling", {}))
    required = ("source", "output_dir", "n_points")
    missing = [key for key in required if key not in sampling and getattr(args, key) is None]
    if missing:
        raise ValueError(
            "Sampling recipe must define " + ", ".join(missing) + "."
        )
    profile = config["cluster"]["bo"]
    partition = args.partition or profile["partition"]
    qos = args.qos or profile["qos"]
    cores = args.cores or profile["cores"]
    time_limit = args.time or profile["time"]
    env_lines = config["cluster"].get("env_setup", [])
    if isinstance(env_lines, str):
        env_block = env_lines
    else:
        env_block = "\n".join(env_lines)
    mem = str(profile.get("mem", "")).strip()
    mem_line = f"#SBATCH --mem={mem}" if mem not in {"", "0", "None"} else ""
    dependency_line = (
        f"#SBATCH --dependency={args.dependency}" if args.dependency else ""
    )
    source = args.source or sampling["source"]
    output_dir = args.output_dir or sampling["output_dir"]
    n_points = args.n_points or int(sampling["n_points"])
    elite_centers = args.elite_centers or int(sampling.get("elite_centers", 48))
    center_selection = args.center_selection or sampling.get(
        "center_selection", "top"
    )
    center_pool_multiplier = args.center_pool_multiplier or int(
        sampling.get("center_pool_multiplier", 5)
    )
    radii = args.radii or sampling.get("radii", [0.025, 0.05, 0.10, 0.20])
    global_fraction = (
        args.global_fraction if args.global_fraction is not None
        else float(sampling.get("global_fraction", 0.50))
    )
    seeds = args.seeds or sampling.get("seeds", [101, 202, 303])
    design_seed = args.design_seed or int(sampling.get("design_seed", 20260713))
    max_workers = args.max_workers or int(
        sampling.get("max_workers", config["parallel"]["max_workers"])
    )
    keep_work = bool(args.keep_work or sampling.get("keep_work", False))
    job_name = args.job_name or sampling.get(
        "job_name", f"btah_sample_{n_points}"
    )
    recipe_path = str(args.config).replace("\\\\", "/")
    command = (
        "python -u -m engine.local_sampling "
        f"--config {shlex.quote(recipe_path)} --source {shlex.quote(str(source))} "
        f"--output-dir {shlex.quote(str(output_dir))} --n-points {n_points} "
        f"--elite-centers {elite_centers} --radii {' '.join(map(str, radii))} "
        f"--center-selection {center_selection} "
        f"--center-pool-multiplier {center_pool_multiplier} "
        f"--global-fraction {global_fraction} --seeds {' '.join(map(str, seeds))} "
        f"--design-seed {design_seed} --max-workers {max_workers}"
    )
    if keep_work:
        command += " --keep-work"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/local_sampling_%j.out
#SBATCH --error=logs/local_sampling_%j.err
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={profile.get("nodes", 1)}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --time={time_limit}
{mem_line}
{dependency_line}

set -euo pipefail
{env_block}
mkdir -p logs
echo "Starting local sampling on $(hostname) at $(date)"
{command}
echo "Finished local sampling at $(date)"
"""
    if args.dry_run:
        print(script)
        return
    Path("logs").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="btah_local_"
    ) as handle:
        handle.write(script)
        temp_path = Path(handle.name)
    try:
        result = subprocess.run(
            ["sbatch", str(temp_path)], check=True, capture_output=True, text=True
        )
        print(result.stdout.strip())
    finally:
        destination = Path("logs") / "last_local_sampling.sh"
        if temp_path.exists():
            shutil.move(str(temp_path), destination)


if __name__ == "__main__":
    main()
