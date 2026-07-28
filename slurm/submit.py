#!/usr/bin/env python3
"""
slurm/submit.py  |  Unified SLURM Job Submission  |  v8

Reads cluster settings from the generated runtime configuration and creates
a SLURM script, then submits it with sbatch.

Usage:
  python slurm/submit.py bo               # submit BO job
  python slurm/submit.py nn               # submit NN training job
  python slurm/submit.py al               # submit Active Learning job
  python slurm/submit.py bo --dry-run     # print generated script without submitting
  python slurm/submit.py bo --config runs/.../_configs/project_cluster.json

Job scripts generated:
  bo  -> python -m engine.run                (+ optional --resume)
  nn  -> python -m engine.nn_surrogate       (+ optional --no-validate)
  al  -> python -m engine.active_learning    (+ optional --no-validate)
       + python -m engine.run --dry-run --save-traj  (post-AL validation)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.config_loader import load_config as load_expanded_config


# ============================================================================
# Script templates
# ============================================================================

def _job_commands(job: str, extra_args: str, config: str = "",
                  al_output_dir: str = "active_learning_output") -> str:
    """Return the shell commands that run inside the SLURM job."""
    if job == "bo":
        if "--resume" in extra_args.split():
            return f"""\
echo "Launching BO resume"
python -u -m engine.run --config {config} {extra_args}
"""
        return f"""\
echo "Pre-flight: dry-run with --no-mpi"
python -u -m engine.run --config {config} --dry-run --no-mpi
echo "Launching BO"
python -u -m engine.run --config {config} {extra_args}
"""
    elif job == "nn":
        return f"""\
echo "Training NN surrogate"
python -u -m engine.nn_surrogate --config {config} {extra_args}
"""
    elif job == "al":
        return f"""\
echo "Active Learning"
python -u -m engine.active_learning --config {config} --output-dir {al_output_dir} {extra_args}
echo "Post-AL dry-run validation"
python -u -m engine.run --config {config} --dry-run --save-traj
"""
    raise ValueError(f"Unknown job type: {job}")


def build_script(job: str, cluster_cfg: dict, extra_args: str,
                 log_dir: str = "logs", config: str = "",
                 al_output_dir: str = "active_learning_output") -> str:
    """
    Generate a complete SLURM batch script string.

    Parameters
    ----------
    job         : "bo" | "nn" | "al"
    cluster_cfg : dict from config.yaml cluster.{job}
    extra_args  : additional CLI flags passed to the Python command
    log_dir     : directory for SLURM stdout/stderr logs
    """
    partition = cluster_cfg["partition"]
    qos       = cluster_cfg["qos"]
    nodes     = cluster_cfg.get("nodes", 1)
    cores     = cluster_cfg["cores"]
    time_str  = cluster_cfg["time"]
    mem       = cluster_cfg["mem"]
    gpu       = int(cluster_cfg.get("gpu", 0))

    gpu_line = f"#SBATCH --gres=gpu:{gpu}" if gpu > 0 else ""
    # mem "0"/empty -> omit the line (some schedulers reject --mem=0; default = all)
    mem_line = f"#SBATCH --mem={mem}" if str(mem).strip() not in ("0", "", "None") else ""

    # Environment setup is cluster-specific. Read it from config.yaml
    # (cluster.{job}.env_setup) so users never edit this file. Accepts a list
    # of shell lines or a single multi-line string. Falls back to the generic
    # module-load template only if env_setup is absent.
    env_setup = cluster_cfg.get("env_setup")
    if env_setup is None:
        load_cuda = "module load cuda/12.1" if gpu > 0 else ""
        env_block = "\n".join(filter(None, [
            "module purge",
            "module load python/3.10",
            "module load mpi/2021.15",
            load_cuda,
            'source "${HOME}/venv/ff_opt/bin/activate"',
        ]))
    elif isinstance(env_setup, (list, tuple)):
        env_block = "\n".join(str(line) for line in env_setup)
    else:
        env_block = str(env_setup)

    commands = _job_commands(job, extra_args, config, al_output_dir)

    script = f"""\
#!/bin/bash
#SBATCH --job-name={job}_v8
#SBATCH --output={log_dir}/{job}_%j.out
#SBATCH --error={log_dir}/{job}_%j.err
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --time={time_str}
{mem_line}
{gpu_line}

set -euo pipefail

# Environment
{env_block}

mkdir -p {log_dir}

echo "=============================="
echo " Job : {job}_v8"
echo " ID  : ${{SLURM_JOB_ID:-local}}"
echo " Node: $(hostname)"
echo " Time: $(date)"
echo "=============================="

# Commands
{commands}
echo "=============================="
echo " Done: $(date)"
echo "=============================="
"""
    return script.strip()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified SLURM job submission for material_v8 pipeline")
    parser.add_argument("job", choices=["bo", "nn", "al"],
                        help="Job to submit: bo | nn | al")
    parser.add_argument(
        "--config", required=True,
        help="Path to the generated internal JSON configuration.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print generated SLURM script without submitting")
    parser.add_argument("--resume",    action="store_true",
                        help="(bo/al) Continue from the latest saved progress")
    parser.add_argument("--bo-method", choices=["auto", "gp", "turbo", "saasbo"],
                        default=None,
                        help="(bo) Override optimization.method")
    parser.add_argument("--model",
                        choices=["mlp_ensemble", "gp", "random_forest", "xgboost"],
                        default=None,
                        help="(nn/al) Override nn.model")
    parser.add_argument("--train-top-fraction", type=float, default=None,
                        help="(nn/al) Override nn.train_top_fraction")
    parser.add_argument("--bo-dir", default=None,
                        help="(nn/al) Explicit BO results directory")
    parser.add_argument("--nn-dir", default=None,
                        help="(al) Initial forward_nn.pt or focused-hybrid directory")
    parser.add_argument("--output-dir", default=None,
                        help="(nn/al) Explicit output directory")
    parser.add_argument(
        "--additional-core-file", action="append", default=None,
        help="(nn) Additional stability-mean CSV; repeat for multiple files",
    )
    parser.add_argument("--core-objective-max", type=float, default=None,
                        help="(nn) Override stable-core objective cutoff")
    parser.add_argument("--buffer-objective-max", type=float, default=None,
                        help="(nn) Override raw-BO buffer objective cutoff")
    parser.add_argument("--core-weight", type=int, default=None,
                        help="(nn) Override stable-core integer weight")
    parser.add_argument("--no-validate", action="store_true",
                        help="(nn/al) Add --no-validate flag")
    parser.add_argument("--save-traj", action="store_true",
                        help="Add --save-traj flag")
    parser.add_argument("--log-dir",   default="logs",
                        help="SLURM log directory (default: logs/)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    config = load_expanded_config(config_path)

    if "cluster" not in config:
        print("ERROR: 'cluster' section missing from config YAML (Section 14).")
        sys.exit(1)

    if args.job not in config["cluster"]:
        print(f"ERROR: cluster.{args.job} not found in config YAML.")
        sys.exit(1)

    # Copy so we can layer the shared env_setup without mutating loaded config.
    cluster_cfg = dict(config["cluster"][args.job])

    # Resolve environment setup
    # Precedence: per-job cluster.{job}.env_setup  >  global cluster.env_setup.
    # The global block is the single place to edit env (conda/modules) for all
    # jobs; a job may override it (e.g. nn needs a cuda module).
    if "env_setup" not in cluster_cfg and "env_setup" in config["cluster"]:
        cluster_cfg["env_setup"] = config["cluster"]["env_setup"]

    # Build extra args
    extra_parts = []
    if args.resume      and args.job in ("bo", "al"):
        extra_parts.append("--resume")
    if args.bo_method   and args.job == "bo":
        extra_parts.extend(["--bo-method", args.bo_method])
    if args.model       and args.job in ("nn", "al"):
        extra_parts.extend(["--model", args.model])
    if args.train_top_fraction is not None and args.job in ("nn", "al"):
        extra_parts.extend(["--train-top-fraction", str(args.train_top_fraction)])
    if args.bo_dir and args.job in ("nn", "al"):
        extra_parts.extend(["--bo-dir", args.bo_dir])
    if args.nn_dir and args.job == "al":
        extra_parts.extend(["--nn-dir", args.nn_dir])
    if args.output_dir and args.job == "bo":
        extra_parts.extend(["--output-dir", args.output_dir])
    if args.output_dir and args.job == "nn":
        extra_parts.extend(["--output-dir", args.output_dir])
    if args.additional_core_file and args.job in ("nn", "al"):
        for path in args.additional_core_file:
            extra_parts.extend(["--additional-core-file", path])
    if args.core_objective_max is not None and args.job in ("nn", "al"):
        extra_parts.extend(["--core-objective-max", str(args.core_objective_max)])
    if args.buffer_objective_max is not None and args.job in ("nn", "al"):
        extra_parts.extend(["--buffer-objective-max", str(args.buffer_objective_max)])
    if args.core_weight is not None and args.job in ("nn", "al"):
        extra_parts.extend(["--core-weight", str(args.core_weight)])
    if args.no_validate and args.job in ("nn", "al"): extra_parts.append("--no-validate")
    if args.save_traj:  extra_parts.append("--save-traj")
    extra_args = " ".join(extra_parts)

    # Generate script
    sys_name = config.get("manifest", {}).get("system_name", "output")
    al_output_dir = args.output_dir or f"active_learning_output_{sys_name}"
    script = build_script(args.job, cluster_cfg, extra_args, args.log_dir,
                          args.config, al_output_dir)

    if args.dry_run:
        print("=" * 60)
        print(f"Generated SLURM script for job={args.job}  (--dry-run, not submitted)")
        print("=" * 60)
        print(script)
        return

    # Submit
    os.makedirs(args.log_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                     delete=False, prefix=f"slurm_{args.job}_") as tmp:
        tmp.write(script + "\n")
        tmp_path = tmp.name

    print(f"Generated: {tmp_path}")
    print(f"Partition : {cluster_cfg['partition']}")
    print(f"QOS       : {cluster_cfg['qos']}")
    print(f"Cores     : {cluster_cfg['cores']}")
    print(f"Time      : {cluster_cfg['time']}")
    print(f"GPU       : {cluster_cfg.get('gpu', 0)}")

    try:
        result = subprocess.run(
            ["sbatch", tmp_path],
            capture_output=True, text=True, check=True)
        print(f"\nSubmitted: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\nWARNING: sbatch not found. Are you on a SLURM cluster?")
        print(f"Script saved at: {tmp_path}")
        print("Submit manually with:  sbatch", tmp_path)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: sbatch failed:\n{e.stderr}")
        sys.exit(1)
    finally:
        # Keep script for reference. Use shutil.move (not os.replace): the temp
        # file lives in /tmp (often a separate tmpfs device), and os.replace =
        # rename(2) fails across devices with "Invalid cross-device link".
        dest = os.path.join(args.log_dir, f"last_{args.job}.sh")
        os.makedirs(args.log_dir, exist_ok=True)
        if os.path.exists(tmp_path):
            shutil.move(tmp_path, dest)
            print(f"Script saved: {dest}")


if __name__ == "__main__":
    main()
# end of submit.py
