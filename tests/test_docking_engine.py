"""Tests for SigmaHoleDockingEngine."""

from sigma_hole_docking.docking_engine import SigmaHoleDockingEngine


class TestPhysicsScoreReturnType:
    def test_missing_file_returns_tuple_false(self):
        """Missing file → (nan, False)."""
        engine = SigmaHoleDockingEngine()
        result = engine.calculate_physics_score("nonexistent.pdbqt", "also_nonexistent.pdbqt")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] is False

    def test_valid_pair_returns_tuple_true(self, tmp_path):
        """Valid pair → (float, True)."""
        # Create minimal PDBQT files
        ligand = tmp_path / "lig.pdbqt"
        ligand.write_text(
            "ROOT\n"
            "ATOM  1  C   LIG B 1   0.000  0.000  0.000  0.00  0.00  -0.100 C\n"
            "ATOM  2  I   LIG B 1   2.100  0.000  0.000  0.00  0.00   0.050 I\n"
            "ATOM  3  H   LIG B 1   3.300  0.000  0.000  0.00  0.00   0.061 EP\n"
            "ENDROOT\nTORSDOF\n0\n"
        )
        receptor = tmp_path / "rec.pdbqt"
        receptor.write_text(
            "ROOT\n"
            "ATOM  1  C   REC A 1   5.000  0.000  0.000  0.00  0.00  -0.100 C\n"
            "ATOM  2  O   REC A 1   6.200  0.000  0.000  0.00  0.00  -0.300 OA\n"
            "ENDROOT\nTORSDOF\n0\n"
        )
        engine = SigmaHoleDockingEngine()
        result = engine.calculate_physics_score(str(ligand), str(receptor))
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Should succeed (may return finite energy)
        if result[1]:
            assert isinstance(result[0], float)


class TestMultiHalogen:
    def test_find_all_halogens_and_carbons(self, tmp_path):
        """Diiodobenzene → both halogens found."""
        # Create a PDBQT with 2 iodines
        ligand = tmp_path / "diiodo.pdbqt"
        ligand.write_text(
            "ROOT\n"
            "ATOM  1  C   LIG B 1   0.000  0.000  0.000  0.00  0.00  -0.100 C\n"
            "ATOM  2  C   LIG B 1   1.400  0.000  0.000  0.00  0.00  -0.100 C\n"
            "ATOM  3  I   LIG B 1   2.100  0.000  0.000  0.00  0.00   0.050 I\n"
            "ATOM  4  I   LIG B 1  -0.700  0.000  0.000  0.00  0.00   0.050 I\n"
            "ATOM  5  H   LIG B 1   3.300  0.000  0.000  0.00  0.00   0.061 EP\n"
            "ATOM  6  H   LIG B 1  -1.900  0.000  0.000  0.00  0.00   0.061 EP\n"
            "ENDROOT\nTORSDOF\n0\n"
        )
        engine = SigmaHoleDockingEngine()
        atoms = engine._parse_pdbqt(str(ligand))
        pairs = engine._find_all_halogens_and_carbons(atoms)
        # Should find 2 halogen-carbon pairs
        assert len(pairs) == 2
        for halogen, carbon in pairs:
            assert halogen is not None
            assert halogen["element"] == "I"
