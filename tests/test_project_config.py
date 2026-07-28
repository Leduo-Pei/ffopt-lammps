from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from workflow.cli import _free_parameter_count
from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file
from workflow.machine import build_machine_profile, load_machine_profile, save_machine_profile
from workflow.project import compose_config, load_project, write_generated_config


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
        cluster = compose_config(self.project, "cluster")
        self.assertEqual(local["machine"]["backend"], "local")
        self.assertEqual(local["lammps"]["executable"], "lmp")
        self.assertEqual(cluster["machine"]["backend"], "slurm")
        self.assertEqual(cluster["cluster"]["bo"]["cores"], 1)

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

    def test_two_node_slurm_profile_distributes_lammps_but_not_nn(self) -> None:
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
        self.assertNotIn("gpu", profile["cluster"]["nn"])
        self.assertEqual(profile["cluster"]["validate"]["tasks"], 1)

    def test_generated_json_config_is_reloadable(self) -> None:
        path = write_generated_config(self.project, "local")
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".json")
        text = path.read_text(encoding="utf-8")
        config = compose_config(self.project, "local")
        self.assertIn(f'"system_name": "{config["manifest"]["system_name"]}"', text)
        self.assertNotIn("_project_path", text)

    def test_parameter_regimes_share_one_schema(self) -> None:
        charge = load_project(ROOT / "examples/btah/charge_only.in")
        full = load_project(ROOT / "examples/btah/full.in")
        self.assertEqual(_free_parameter_count(compose_config(charge, "cluster")), 13)
        self.assertEqual(_free_parameter_count(compose_config(full, "cluster")), 41)

        fix_sigma = parse_input_file(ROOT / "examples/btah/full.in")
        fix_sigma.parameters.fixed.add("sigma")
        self.assertEqual(_free_parameter_count(compile_input(fix_sigma).config), 27)

    def test_charge_only_uses_general_ann_cutoffs(self) -> None:
        config = compose_config(self.project, "cluster")
        training = config["nn"]["training_data"]
        self.assertEqual(training["core_objective_max"], 1.0)
        self.assertEqual(training["buffer_objective_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
