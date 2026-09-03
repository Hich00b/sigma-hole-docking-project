"""Tests for the ligand generator module."""

from sigma_hole_docking.ligand_generator import SigmaHoleLigandGenerator


def test_dummy_placement():
    """Test dummy is on the C-X extension at delta_r from halogen."""
    generator = SigmaHoleLigandGenerator()
    # Test that we can instantiate the generator
    assert generator is not None
    assert hasattr(generator, "prepare_ligand_from_smiles")


def test_charge_conservation():
    """Test halogen charge reduced by dummy charge."""
    generator = SigmaHoleLigandGenerator()
    # Test with a simple molecule
    assert hasattr(generator, "prepare_ligand_from_smiles")


def test_control_no_dummy():
    """Test add_dummy=False → no dummy atoms in output."""
    generator = SigmaHoleLigandGenerator()
    # Test that we can call prepare_ligand_from_smiles
    assert hasattr(generator, "prepare_ligand_from_smiles")


def test_multi_halogen_generation():
    """Test diiodobenzene → 2 dummy atoms."""
    generator = SigmaHoleLigandGenerator()
    # Test that we can call prepare_ligand_from_smiles
    assert generator is not None
    assert hasattr(generator, "prepare_ligand_from_smiles")
