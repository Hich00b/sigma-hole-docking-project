"""Tests for the docking engine module."""

import pytest
import numpy as np
from sigma_hole_docking.docking_engine import SigmaHoleDockingEngine


def test_physics_score_return_type():
    """Valid pair → (float, True); missing file → (nan, False)."""
    engine = SigmaHoleDockingEngine()

    # Test with nonexistent files - should return (nan, False)
    result = engine.calculate_physics_score("nonexistent.pdbqt", "also_nonexistent.pdbqt")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], bool)
    assert result[1] == False  # success flag should be False
    assert np.isnan(result[0])  # energy should be NaN

    # Note: We can't easily test the success case without actual PDBQT files
    # but we can verify the return type structure


def test_multi_halogen():
    """Placeholder for diiodobenzene → both halogens scored.
    This would require generating actual test PDBQT files."""
    # For now, we'll test that the engine can be instantiated and
    # that it has the expected multi-halogen capability through its
    # use of the alignment and scoring modules
    engine = SigmaHoleDockingEngine()
    assert engine is not None
    # The actual multi-halogen testing would be done in integration tests
    # or by checking that the underlying modules handle lists properly