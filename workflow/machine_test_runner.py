"""Distributed SLURM machine acceptance used by ``ffopt machine test``."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .slurm_pool import SlurmCommandPool


def audit_worker_affinities(
    records: list[dict], *, expected_workers: int, cpus_per_worker: int
) -> list[str]:
    """Return isolation errors for a persistent SLURM worker allocation."""
    errors: list[str] = []
    if len(records) != expected_workers:
        errors.append(
            f"expected {expected_workers} ready workers, found {len(records)}"
        )
    by_host: dict[str, list[tuple[int, set[int]]]] = {}
    for record in records:
        rank = int(record.get("rank", -1))
        host = str(record.get("hostname", "")).strip() or "<unknown>"
        try:
            cpus = {int(value) for value in record.get("cpus", [])}
        except (TypeError, ValueError):
            cpus = set()
        if len(cpus) < cpus_per_worker:
            errors.append(
                f"worker {rank} on {host} has {len(cpus)} CPUs; "
                f"requires at least {cpus_per_worker}"
            )
        for other_rank, other_cpus in by_host.setdefault(host, []):
            overlap = sorted(cpus & other_cpus)
            if overlap:
                errors.append(
                    f"workers {other_rank} and {rank} on {host} share CPUs {overlap}"
                )
        by_host[host].append((rank, cpus))
    return errors


def run_distributed_test(
    *,
    root: Path,
    lammps: str,
    mpi: str,
    mpi_flavor: str | None,
    input_file: Path,
    workers: int,
    nodes: int,
    ranks: int,
    omp_threads: int,
) -> tuple[bool, list[str], list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    pool = SlurmCommandPool(
        launcher="srun",
        root=root / "pool",
        workers=workers,
        nodes=nodes,
        cpus_per_worker=ranks * omp_threads,
    )
    pool.start()
    try:
        ready = sorted((pool.root / "ready").glob("worker_*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in ready]
        hostnames = sorted({
            str(record["hostname"]) for record in records
        })
        affinity_errors = audit_worker_affinities(
            records,
            expected_workers=workers,
            cpus_per_worker=ranks * omp_threads,
        )
        results = []
        if not affinity_errors:
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for index in range(workers):
                    work_dir = root / f"worker_{index:04d}"
                    work_dir.mkdir(exist_ok=True)
                    future = executor.submit(
                        pool.run,
                        [lammps, "-in", str(input_file)],
                        cwd=str(work_dir),
                        env={"OMP_NUM_THREADS": str(omp_threads)},
                        timeout=300,
                        mpi_launcher=mpi if ranks > 1 else None,
                        mpi_launcher_flavor=mpi_flavor if ranks > 1 else None,
                        mpi_ranks=ranks,
                    )
                    futures[future] = index
                results = [future.result() for future in as_completed(futures)]
    finally:
        pool.close()

    all_lammps_passed = len(results) == workers and all(
        result.returncode == 0 and "FFOPT_MACHINE_TEST_OK" in result.stdout
        for result in results
    )
    return (
        all_lammps_passed and not affinity_errors and len(hostnames) >= nodes,
        hostnames,
        affinity_errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--lammps", required=True)
    parser.add_argument("--mpi", required=True)
    parser.add_argument("--mpi-flavor", choices=("openmpi", "intelmpi"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--nodes", required=True, type=int)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--omp-threads", required=True, type=int)
    args = parser.parse_args()
    job_key = os.environ.get("SLURM_JOB_ID", f"process_{os.getpid()}")
    ok, hostnames, affinity_errors = run_distributed_test(
        root=args.root.resolve() / f"job_{job_key}",
        lammps=args.lammps,
        mpi=args.mpi,
        mpi_flavor=args.mpi_flavor,
        input_file=args.input.resolve(),
        workers=args.workers,
        nodes=args.nodes,
        ranks=args.ranks,
        omp_threads=args.omp_threads,
    )
    print(f"FFOpt distributed machine-test hosts: {', '.join(hostnames)}")
    for error in affinity_errors:
        print(f"FFOpt worker-affinity error: {error}")
    if not ok:
        raise SystemExit("Distributed LAMMPS machine test failed")
    print("FFOPT_MACHINE_TEST_OK")


if __name__ == "__main__":
    main()
