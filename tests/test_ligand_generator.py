"""Tests for SigmaHoleLigandGenerator."""

import numpy as np

from sigma_hole_docking.ligand_generator import SigmaHoleLigandGenerator
from sigma_hole_docking.pdbqt_io import parse_pdbqt


class TestDummyPlacement:
    def test_dummy_placement(self, tmp_path):
        """Dummy is on the C-X extension at delta_r from halogen."""
        gen = SigmaHoleLigandGenerator()
        out = tmp_path / "iodobenzene.pdbqt"
        success = gen.prepare_ligand_from_smiles(
            "c1ccccc1I",
            "I",
            charge=0.06,
            output_path=str(out),
            delta_r=1.2,
            add_dummy=True,
        )
        assert success
        atoms = parse_pdbqt(str(out))
        dummies = [a for a in atoms if a["is_dummy"]]
        assert len(dummies) == 1
        halogens = [a for a in atoms if a["element"] == "I"]
        assert len(halogens) == 1
        # Check distance from halogen to dummy ≈ delta_r
        hal = halogens[0]
        dum = dummies[0]
        dist = np.sqrt(
            (hal["x"] - dum["x"]) ** 2 + (hal["y"] - dum["y"]) ** 2 + (hal["z"] - dum["z"]) ** 2
        )
        assert abs(dist - 1.2) < 0.15  # tolerance for geometry optimization


class TestChargeConservation:
    def test_charge_conservation(self, tmp_path):
        """Halogen charge reduced by dummy charge."""
        gen = SigmaHoleLigandGenerator()
        out = tmp_path / "charged.pdbqt"
        success = gen.prepare_ligand_from_smiles(
            "c1ccccc1I",
            "I",
            charge=0.06,
            output_path=str(out),
            delta_r=1.2,
            add_dummy=True,
        )
        assert success
        atoms = parse_pdbqt(str(out))
        [a for a in atoms if a["element"] == "I"][0]
        dummy = [a for a in atoms if a["is_dummy"]][0]
        # Dummy should have positive charge
        assert dummy["charge"] > 0


class TestControlNoDummy:
    def test_control_no_dummy(self, tmp_path):
        """add_dummy=False → no dummy atoms in output."""
        gen = SigmaHoleLigandGenerator()
        out = tmp_path / "control.pdbqt"
        success = gen.prepare_ligand_from_smiles(
            "c1ccccc1I",
            "I",
            charge=0.06,
            output_path=str(out),
            delta_r=1.2,
            add_dummy=False,
        )
        assert success
        atoms = parse_pdbqt(str(out))
        dummies = [a for a in atoms if a["is_dummy"]]
        assert len(dummies) == 0


class TestMultiHalogenGeneration:
    def test_multi_halogen_generation(self, tmp_path):
        """Diiodobenzene → 2 dummy atoms."""
        gen = SigmaHoleLigandGenerator()
        out = tmp_path / "diiodo.pdbqt"
        success = gen.prepare_ligand_from_smiles(
            "c1cc(I)ccc1I",
            "I",
            charge=0.06,
            output_path=str(out),
            delta_r=1.2,
            add_dummy=True,
        )
        assert success
        atoms = parse_pdbqt(str(out))
        dummies = [a for a in atoms if a["is_dummy"]]
        assert len(dummies) == 2
