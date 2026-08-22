"""Tests for consolidated PDBQT I/O."""

from sigma_hole_docking.pdbqt_io import (
    parse_pdbqt,
    write_pdbqt,
    normalize_element,
    is_dummy_atom,
)


class TestNormalizeElement:
    def test_cl(self):
        assert normalize_element("CL") == "Cl"

    def test_br(self):
        assert normalize_element("BR") == "Br"

    def test_i(self):
        assert normalize_element("I") == "I"

    def test_carbon(self):
        assert normalize_element("C") == "C"


class TestIsDummyAtom:
    def test_ep_type(self):
        assert is_dummy_atom("EP", "H") is True

    def test_normal_atom(self):
        assert is_dummy_atom("C", "C") is False


class TestParsePdbqt:
    def test_round_trip(self, tmp_path):
        """Write a small set of atoms → parse back → count and charges match."""
        atoms = [
            {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.1, "atom_type": "C"},
            {"element": "O", "x": 1.2, "y": 0.0, "z": 0.0, "charge": -0.3, "atom_type": "OA"},
            {"element": "I", "x": 2.5, "y": 0.0, "z": 0.0, "charge": 0.05, "atom_type": "I"},
        ]
        path = tmp_path / "test.pdbqt"
        write_pdbqt(atoms, str(path), remarks=["Test file"])
        parsed = parse_pdbqt(str(path))
        assert len(parsed) == 3
        assert parsed[0]["element"] == "C"
        assert abs(parsed[0]["charge"] - (-0.1)) < 1e-4
        assert parsed[1]["element"] == "O"
        assert abs(parsed[1]["x"] - 1.2) < 1e-3

    def test_dummy_detection(self, tmp_path):
        """Dummy atom (EP type) flagged is_dummy=True."""
        atoms = [
            {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0, "charge": 0.0, "atom_type": "C"},
            {"element": "H", "x": 1.0, "y": 0.0, "z": 0.0, "charge": 0.05, "atom_type": "EP"},
        ]
        path = tmp_path / "dummy.pdbqt"
        write_pdbqt(atoms, str(path))
        parsed = parse_pdbqt(str(path))
        assert parsed[0]["is_dummy"] is False
        assert parsed[1]["is_dummy"] is True

    def test_element_normalization(self, tmp_path):
        """CL → Cl, BR → Br in parsed output."""
        atoms = [
            {"element": "CL", "x": 0.0, "y": 0.0, "z": 0.0, "charge": 0.0, "atom_type": "CL"},
            {"element": "BR", "x": 1.0, "y": 0.0, "z": 0.0, "charge": 0.0, "atom_type": "BR"},
        ]
        path = tmp_path / "norm.pdbqt"
        write_pdbqt(atoms, str(path))
        parsed = parse_pdbqt(str(path))
        assert parsed[0]["element"] == "Cl"
        assert parsed[1]["element"] == "Br"

    def test_malformed_line(self, tmp_path):
        """Bad line skipped, no crash."""
        path = tmp_path / "bad.pdbqt"
        path.write_text(
            "ROOT\n"
            "ATOM 1 C LIG B 1 0.0 0.0 0.0 0.00 0.00 -0.1 C\n"
            "ATOM bad line here\n"
            "ATOM 2 O LIG B 1 1.0 0.0 0.0 0.00 0.00 -0.3 OA\n"
            "ENDROOT\nTORSDOF\n0\n"
        )
        parsed = parse_pdbqt(str(path))
        assert len(parsed) == 2  # only the two valid lines

    def test_missing_file(self):
        """Missing file returns empty list, no exception."""
        atoms = parse_pdbqt("nonexistent_file.pdbqt")
        assert atoms == []
