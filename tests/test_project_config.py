from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from engine.lammps_interface import LAMMPSRunner
from workflow.cli import _free_parameter_count
from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file
from workflow.machine import build_machine_profile, load_machine_profile, save_machine_profile
from workflow.project import compose_config, load_project


ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_config = tempfile.TemporaryDirectory()
        self.previous_config_dir = os.environ.get("FFOPT_CONFIG_DIR")
        os.environ["FFOPT_CONFIG_DIR"] = self.temp_config.name
        self.project = load_project(ROOT / "examples/btah/charge_only.in")
        self.project.data["project"]["run_root"] = str(
            Path(self.temp_config.name) / "runs" / self.project.name
        )

    def tearDown(self) -> None:
        if self.previous_config_dir is None:
            os.environ.pop("FFOPT_CONFIG_DIR", None)
        else:
            os.environ["FFOPT_CONFIG_DIR"] = self.previous_config_dir
        self.temp_config.cleanup()

    def test_active_input_matches_declared_parameter_regime(self) -> None:
        config = compose_config(self.project, "local")
        self.assertEqual(_free_parameter_count(config), 13)
        self.assertEqual(config["targets"]["esub_proxy"]["value"], 98.5)
        self.assertFalse(config["adsorption"]["enabled"])
        self.assertTrue(
            config["validation"]["property_settings"]["adsorption"]["enabled"]
        )

    def test_machine_profile_is_selected_outside_scientific_input(self) -> None:
        local = compose_config(self.project, "local")
        self.assertEqual(local["machine"]["backend"], "local")
        self.assertEqual(local["lammps"]["executable"], "lmp")
        profile = build_machine_profile(
            name="cluster-test", backend="slurm", workers=1, ranks=1, cores=1
        )
        save_machine_profile("cluster-test", profile)
        cluster = compose_config(self.project, "cluster-test")
        self.assertEqual(cluster["machine"]["backend"], "slurm")
        self.assertEqual(cluster["cluster"]["bo"]["cores"], 1)

    def test_unconfigured_cluster_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only 'local' has a built-in profile"):
            compose_config(self.project, "cluster")

    def test_user_machine_profile_uses_single_toml_file(self) -> None:
        profile = build_machine_profile(
            name="workstation",
            backend="local",
            lammps="D:/LAMMPS/bin/lmp.exe",
            mpi="D:/MPI/mpiexec.exe",
            workers=8,
            ranks=4,
        )
        path = save_machine_profile("workstation", profile)
        self.assertEqual(path.name, "machines.toml")
        loaded = load_machine_profile("workstation")
        self.assertIsNotNone(loaded)
        config = compose_config(self.project, "workstation")
        self.assertEqual(config["lammps"]["executable"], "D:/LAMMPS/bin/lmp.exe")
        self.assertEqual(config["parallel"]["max_workers"], 8)
        self.assertEqual(config["parallel"]["cores_per_worker"], 4)

    def test_input_paths_are_relative_to_input_file(self) -> None:
        self.assertEqual(self.project.root, (ROOT / "examples/btah").resolve())
        config = compose_config(self.project, "local")
        self.assertEqual(
            Path(config["manifest"]["data_files"]["bulk"]),
            (ROOT / "data/bulk/BTAH_822_bulk.data").resolve(),
        )

    def test_two_node_slurm_profile_keeps_nn_workers_on_one_node(self) -> None:
        profile = build_machine_profile(
            name="ccelab-2node",
            backend="slurm",
            lammps="/opt/ffopt/bin/lmp",
            mpi="/opt/ffopt/bin/mpirun",
            workers=24,
            ranks=4,
            omp_threads=1,
            nodes=2,
            cores=96,
            partition="CPU",
            memory_per_node="64G",
        )
        self.assertEqual(profile["cluster"]["bo"]["nodes"], 2)
        self.assertEqual(profile["cluster"]["bo"]["tasks"], 24)
        self.assertEqual(profile["cluster"]["bo"]["tasks_per_node"], 12)
        self.assertEqual(profile["cluster"]["bo"]["cpus_per_task"], 4)
        self.assertEqual(profile["cluster"]["bo"]["mem"], "64G")
        self.assertEqual(profile["cluster"]["nn"]["mem"], "64G")
        self.assertTrue(profile["cluster"]["bo"]["distributed_steps"])
        self.assertEqual(profile["parallel"]["scheduler_launcher"], "srun")
        self.assertEqual(profile["parallel"]["workers_per_node"], 12)
        self.assertEqual(profile["cluster"]["nn"]["nodes"], 1)
        self.assertEqual(profile["cluster"]["nn"]["cores"], 48)
        self.assertEqual(profile["cluster"]["nn"]["tasks"], 12)
        self.assertEqual(profile["cluster"]["nn"]["tasks_per_node"], 12)
        self.assertEqual(profile["cluster"]["nn"]["cpus_per_task"], 4)
        self.assertTrue(profile["cluster"]["nn"]["distributed_steps"])
        self.assertNotIn("gpu", profile["cluster"]["nn"])
        self.assertEqual(profile["cluster"]["validate"]["tasks"], 1)

    def test_slurm_profile_is_stored_concisely_and_expanded_when_loaded(self) -> None:
        profile = build_machine_profile(
            name="compact-2node",
            backend="slurm",
            lammps="/opt/ffopt/bin/lmp",
            mpi="/opt/ffopt/bin/mpirun",
            mpi_flavor="openmpi",
            workers=20,
            ranks=4,
            nodes=2,
            cores=80,
            partition="CPU",
            memory_per_node="64G",
            walltime="06:00:00",
            timeout=7200,
        )
        path = save_machine_profile("compact-2node", profile)
        with path.open("rb") as handle:
            stored = tomllib.load(handle)["machines"]["compact-2node"]

        self.assertEqual(stored["format"], 2)
        self.assertEqual(stored["backend"], "slurm")
        self.assertNotIn("cluster", stored)
        self.assertEqual(stored["parallel"]["workers"], 20)
        self.assertEqual(stored["parallel"]["mpi_ranks"], 4)
        self.assertEqual(stored["slurm"]["total_cores"], 80)

        loaded = load_machine_profile("compact-2node")
        self.assertEqual(loaded["cluster"]["bo"]["tasks"], 20)
        self.assertEqual(loaded["cluster"]["nn"]["nodes"], 1)
        self.assertEqual(loaded["cluster"]["validate"]["tasks"], 1)

    def test_legacy_expanded_machine_profile_remains_supported(self) -> None:
        path = Path(self.temp_config.name) / "machines.toml"
        path.write_text(
            """
[machines.legacy.machine]
name = "legacy"
backend = "slurm"

[machines.legacy.cluster.bo]
partition = "CPU"
nodes = 1
cores = 4
tasks = 1
cpus_per_task = 4
""".strip()
            + "\n",
            encoding="utf-8",
        )
        loaded = load_machine_profile("legacy")
        self.assertEqual(loaded["machine"]["backend"], "slurm")
        self.assertEqual(loaded["cluster"]["bo"]["cores"], 4)

    def test_compact_stage_overrides_use_public_field_names(self) -> None:
        path = Path(self.temp_config.name) / "machines.toml"
        path.write_text(
            """
[machines.custom]
format = 2
backend = "slurm"

[machines.custom.lammps]
executable = "/opt/ffopt/bin/lmp"
mpiexec = "/opt/ffopt/bin/mpirun"
mpi_flavor = "openmpi"
timeout = 3600

[machines.custom.parallel]
workers = 4
mpi_ranks = 2
omp_threads = 1

[machines.custom.slurm]
partition = "compute"
nodes = 1
total_cores = 8
walltime = "02:00:00"

[machines.custom.stages.nn]
walltime = "01:00:00"
memory_per_node = "32G"
gpus = 1
""".strip()
            + "\n",
            encoding="utf-8",
        )
        loaded = load_machine_profile("custom")
        self.assertEqual(loaded["cluster"]["nn"]["time"], "01:00:00")
        self.assertEqual(loaded["cluster"]["nn"]["mem"], "32G")
        self.assertEqual(loaded["cluster"]["nn"]["gpu"], 1)

    def test_machine_worker_count_does_not_change_bo_batch_size(self) -> None:
        one = build_machine_profile(
            name="one-node", backend="slurm", workers=12, ranks=4,
            nodes=1, cores=48, partition="CPU",
        )
        two = build_machine_profile(
            name="two-node", backend="slurm", workers=24, ranks=4,
            nodes=2, cores=96, partition="CPU",
        )
        save_machine_profile("one-node", one)
        save_machine_profile("two-node", two)
        one_config = compose_config(self.project, "one-node")
        two_config = compose_config(self.project, "two-node")
        self.assertEqual(one_config["optimization"]["batch_size"], 48)
        self.assertEqual(two_config["optimization"]["batch_size"], 48)
        self.assertEqual(one_config["parallel"]["max_workers"], 12)
        self.assertEqual(two_config["parallel"]["max_workers"], 24)

    def test_composed_config_records_software_provenance(self) -> None:
        config = compose_config(self.project, "local")
        self.assertIn("ffopt_lammps", config["manifest"]["software"])
        self.assertIn("python", config["manifest"]["software"])

    def test_bookkeeping_runner_does_not_start_scheduler_workers(self) -> None:
        config = compose_config(self.project, "local")
        with patch(
            "engine.lammps_interface.SlurmCommandPool.from_config",
            side_effect=AssertionError("scheduler pool must stay disabled"),
        ):
            runner = LAMMPSRunner(config, start_scheduler_pool=False)
        self.assertIsNone(runner.scheduler_pool)

    def test_parameter_regimes_share_one_schema(self) -> None:
        charge = load_project(ROOT / "examples/btah/charge_only.in")
        full = load_project(ROOT / "examples/btah/full.in")
        self.assertEqual(_free_parameter_count(compose_config(charge, "local")), 13)
        self.assertEqual(_free_parameter_count(compose_config(full, "local")), 41)

        fix_sigma = parse_input_file(ROOT / "examples/btah/full.in")
        fix_sigma.parameters.fixed.add("sigma")
        self.assertEqual(_free_parameter_count(compile_input(fix_sigma).config), 27)

    def test_charge_only_uses_general_ann_cutoffs(self) -> None:
        config = compose_config(self.project, "local")
        training = config["nn"]["training_data"]
        self.assertEqual(training["core_objective_max"], 1.0)
        self.assertEqual(training["buffer_objective_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
