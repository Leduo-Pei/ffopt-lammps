from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from workflow.cli import _free_parameter_count
from workflow.project import compose_config, load_project, write_generated_config
from workflow.machine import build_machine_profile, load_machine_profile, save_machine_profile


class ProjectConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_config = tempfile.TemporaryDirectory()
        self.previous_config_dir = os.environ.get("FFOPT_CONFIG_DIR")
        os.environ["FFOPT_CONFIG_DIR"] = self.temp_config.name
        self.project = load_project("projects/project.yaml")

    def tearDown(self) -> None:
        if self.previous_config_dir is None:
            os.environ.pop("FFOPT_CONFIG_DIR", None)
        else:
            os.environ["FFOPT_CONFIG_DIR"] = self.previous_config_dir
        self.temp_config.cleanup()

    def test_active_project_matches_declared_parameter_regime(self) -> None:
        config = compose_config(self.project, "local")
        expected = {
            "btah_full": 41,
            "btah_fix_sigma": 27,
            "btah_charge_only": 13,
        }
        self.assertEqual(_free_parameter_count(config), expected[self.project.name])
        self.assertEqual(config["targets"]["esub_proxy"]["value"], 98.5)
        self.assertFalse(config["adsorption"]["enabled"])

    def test_machine_profile_is_selected_by_one_flag(self) -> None:
        local = compose_config(self.project, "local")
        cluster = compose_config(self.project, "cluster")
        self.assertEqual(local["machine"]["backend"], "local")
        self.assertEqual(local["lammps"]["executable"], "lmp")
        self.assertEqual(cluster["machine"]["backend"], "slurm")
        self.assertEqual(cluster["cluster"]["bo"]["cores"], 1)

    def test_user_machine_profile_overrides_portable_defaults(self) -> None:
        profile = build_machine_profile(
            name="workstation",
            backend="local",
            lammps="D:/LAMMPS/bin/lmp.exe",
            mpi="D:/MPI/mpiexec.exe",
            workers=8,
            ranks=4,
        )
        save_machine_profile("workstation", profile)
        loaded = load_machine_profile("workstation")
        self.assertIsNotNone(loaded)
        config = compose_config(self.project, "workstation")
        self.assertEqual(config["lammps"]["executable"], "D:/LAMMPS/bin/lmp.exe")
        self.assertEqual(config["parallel"]["max_workers"], 8)
        self.assertEqual(config["parallel"]["cores_per_worker"], 4)

    def test_external_project_paths_are_relative_to_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project_file = root / "project.yaml"
            project_file.write_text(
                "project:\n  name: external\n  run_root: outputs\nmodules: {}\n",
                encoding="utf-8",
            )
            project = load_project(project_file)
            self.assertEqual(project.root, root.resolve())
            self.assertEqual(project.run_root, (root / "outputs").resolve())

    def test_generated_config_is_reloadable(self) -> None:
        path = write_generated_config(self.project, "local")
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        config = compose_config(self.project, "local")
        self.assertIn(f"system_name: {config['manifest']['system_name']}", text)
        self.assertNotIn("_config_path", text)

    def test_parameter_regimes_are_isolated_projects(self) -> None:
        expected = {
            "projects/btah_full.yaml": 41,
            "projects/btah_fix_sigma.yaml": 27,
            "projects/btah_charge_only.yaml": 13,
        }
        names = set()
        for path, dimensions in expected.items():
            project = load_project(path)
            names.add(project.name)
            self.assertEqual(_free_parameter_count(compose_config(project, "cluster")), dimensions)
        self.assertEqual(len(names), 3)

    def test_charge_only_does_not_reuse_invalid_nn_cutoffs(self) -> None:
        project = load_project("projects/btah_charge_only.yaml")
        config = compose_config(project, "cluster")
        training = config["nn"]["training_data"]
        self.assertNotEqual(training["core_objective_max"], 0.30)
        self.assertNotEqual(training["buffer_objective_max"], 0.40)


if __name__ == "__main__":
    unittest.main()
