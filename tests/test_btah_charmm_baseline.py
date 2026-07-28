from __future__ import annotations

import math
import unittest
from pathlib import Path

from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file


ROOT = Path(__file__).resolve().parents[1]


def atom_types(config: dict) -> dict[str, dict]:
    return {item["label"]: item for item in config["atom_types"]}


def pair_coefficients(path: Path) -> dict[str, tuple[float, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        index for index, line in enumerate(lines)
        if line.strip().startswith("Pair Coeffs")
    )
    result: dict[str, tuple[float, float]] = {}
    started = False
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        payload, _, comment = stripped.partition("#")
        columns = payload.split()
        if len(columns) < 3 or not columns[0].isdigit():
            if started:
                break
            continue
        started = True
        label = comment.strip()
        if label.startswith("bh"):
            result[label] = (float(columns[1]), float(columns[2]))
    return result


class BtahCharmmBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        full_document = parse_input_file(ROOT / "examples/btah/full.in")
        self.full = atom_types(compile_input(full_document).config)

        fix_sigma_document = parse_input_file(ROOT / "examples/btah/full.in")
        fix_sigma_document.parameters.fixed.add("sigma")
        self.fix_sigma = atom_types(compile_input(fix_sigma_document).config)

        charge_document = parse_input_file(ROOT / "examples/btah/charge_only.in")
        self.charge_only = atom_types(compile_input(charge_document).config)

    def test_reduced_projects_derive_from_full_charmm_reference(self) -> None:
        self.assertEqual(set(self.full), set(self.fix_sigma))
        self.assertEqual(set(self.full), set(self.charge_only))
        for label, reference in self.full.items():
            reference_params = reference["params"]
            epsilon_reference = float(reference_params["epsilon"]["init"])
            sigma_reference = float(reference_params["sigma"]["init"])

            fixed_sigma_params = self.fix_sigma[label]["params"]
            self.assertEqual(
                fixed_sigma_params["epsilon"], reference_params["epsilon"]
            )
            self.assertTrue(math.isclose(
                float(fixed_sigma_params["sigma"]), sigma_reference,
                rel_tol=0.0, abs_tol=1.0e-10,
            ))

            charge_only_params = self.charge_only[label]["params"]
            self.assertTrue(math.isclose(
                float(charge_only_params["epsilon"]), epsilon_reference,
                rel_tol=0.0, abs_tol=1.0e-10,
            ))
            self.assertTrue(math.isclose(
                float(charge_only_params["sigma"]), sigma_reference,
                rel_tol=0.0, abs_tol=1.0e-10,
            ))
            self.assertEqual(
                fixed_sigma_params["charge"], reference_params["charge"]
            )
            self.assertEqual(
                charge_only_params["charge"], reference_params["charge"]
            )

    def test_data_pair_coefficients_match_charmm_reference(self) -> None:
        expected = {
            label: (
                float(item["params"]["epsilon"]["init"]),
                float(item["params"]["sigma"]["init"]),
            )
            for label, item in self.full.items()
        }
        paths = [
            ROOT / "data/bulk/BTAH_822_bulk.data",
            ROOT / "data/molecule/BTAH_822_single.data",
            ROOT / "data/adsorption/ad_complex.data",
            ROOT / "data/adsorption/ad_mol.data",
        ]
        for path in paths:
            with self.subTest(path=path):
                observed = pair_coefficients(path)
                self.assertEqual(set(observed), set(expected))
                for label, values in expected.items():
                    self.assertTrue(math.isclose(
                        observed[label][0], values[0],
                        rel_tol=0.0, abs_tol=1.0e-10,
                    ))
                    self.assertTrue(math.isclose(
                        observed[label][1], values[1],
                        rel_tol=0.0, abs_tol=1.0e-10,
                    ))


if __name__ == "__main__":
    unittest.main()
