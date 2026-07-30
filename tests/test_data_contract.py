from pathlib import Path

from workflow.data_contract import check_data_files, format_data_report

ROOT = Path(__file__).resolve().parents[1]


def test_btah_bulk_and_single_are_compatible():
    report = check_data_files(
        bulk=ROOT / "data/bulk/BTAH_822_bulk.data",
        single=ROOT / "data/molecule/BTAH_822_single.data",
    )
    assert report.ok
    assert not report.errors


def test_btah_adsorption_reorders_types_by_label():
    report = check_data_files(
        bulk=ROOT / "data/bulk/BTAH_822_bulk.data",
        complex=ROOT / "data/adsorption/ad_complex.data",
        slab=ROOT / "data/adsorption/ad_slab.data",
        molecule=ROOT / "data/adsorption/ad_mol.data",
    )
    assert report.ok
    assert not any(item.code == "type_id_mismatch" for item in report.findings)
    assert any(item.code == "initial_charge_mismatch" for item in report.findings)
    assert "Result: PASS" in format_data_report(report)


def test_bulk_single_type_id_mismatch_is_an_error(tmp_path):
    source = (ROOT / "data/molecule/BTAH_822_single.data").read_text(
        encoding="utf-8"
    )
    source = source.replace("1  14.007000 # bhN1", "1   1.008000 # bhN1", 1)
    mismatched = tmp_path / "single.data"
    mismatched.write_text(source, encoding="utf-8")
    report = check_data_files(
        bulk=ROOT / "data/bulk/BTAH_822_bulk.data",
        single=mismatched,
    )
    assert not report.ok
    assert any(item.code == "mass_mismatch" for item in report.errors)
