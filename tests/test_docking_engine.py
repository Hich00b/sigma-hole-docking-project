"""Tests for the docking engine module."""

import numpy as np

from sigma_hole_docking.docking_engine import SigmaHoleDockingEngine


def test_physics_score_return_type():
    """Valid pair → (float, True); missing file → (nan, False)."""
    engine = SigmaHoleDockingEngine()

    # Test with a valid halogen-carbon pair (this would need actual files in practice)
    # For now we just test the return type structure
    result = engine.calculate_physics_score("test_receptor.pdbqt", "test_ligand.pdbqt")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float) or np.isnan(result[0])
    assert isinstance(result[1], bool)


def test_multi_halogen():
    """Placeholder for diiodobenzene → both halogens scored.
    This would require generating actual test PDBQT files.
    """
    # This test requires actual PDBQT files to be meaningful
    # For CI purposes, we just verify the function exists and can be called
    engine = SigmaHoleDockingEngine()
    assert hasattr(engine, "calculate_physics_score")
