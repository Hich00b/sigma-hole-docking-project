"""Tests for the charge calculator module."""

import pytest
import numpy as np
import pandas as pd
from sigma_hole_docking.charge_calculator import SigmaHoleChargeCalculator


def test_charge_formula():
    """Test iodobenzene Vmax=26.0, I → q ≈ 0.0611 e (4 decimals)."""
    calculator = SigmaHoleChargeCalculator()
    # iodobenzene: Vmax=26.0, I: r_iso=1.98, delta_r=1.2
    q = calculator.calculate_charge(26.0, "I")
    assert abs(q - 0.0611) < 1e-4, f"Expected ~0.0611, got {q}"


def test_vmax_zero():
    """Test Vmax=0 → q = 0."""
    calculator = SigmaHoleChargeCalculator()
    q = calculator.calculate_charge(0.0, "I")
    assert q == 0.0


def test_unknown_halogen():
    """Test unknown halogen → raises ValueError."""
    calculator = SigmaHoleChargeCalculator()
    with pytest.raises(ValueError, match="Unknown halogen"):
        calculator.calculate_charge(26.0, "Xx")  # Xx is not a valid halogen


def test_delta_r_exceeds_r_iso():
    """Test delta_r exceeds r_iso → gap clamped to 0.1 Å, warning logged."""
    calculator = SigmaHoleChargeCalculator()
    # For iodine: r_iso = 1.98, if delta_r > 1.98, gap should be clamped to 0.1
    # Let's test with a large delta_r
    q = calculator.calculate_charge(26.0, "I", delta_r=3.0)  # delta_r=3.0 > r_iso=1.98
    # Expected: gap = max(r_iso - delta_r, 0.1) = max(1.98 - 3.0, 0.1) = 0.1
    # q = Vmax * gap / k_coulomb = 26.0 * 0.1 / 332.06 = 0.00783
    expected_q = 26.0 * 0.1 / 332.06
    assert abs(q - expected_q) < 1e-4


def test_batch_dataframe():
    """Test DataFrame in → dummy_charge column added."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({
        'compound_id': ['TEST1', 'TEST2'],
        'halogen': ['Cl', 'Br'],
        'vmax': [20.0, 25.0]
    })
    result_df = calculator.batch_calculate_from_dataframe(df)
    assert 'dummy_charge' in result_df.columns
    assert len(result_df) == 2
    # Check that charges are calculated (not zero)
    assert all(result_df['dummy_charge'] != 0)


def test_save_charges_rename():
    """Test output CSV has dummy_charge_e column."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({
        'compound_id': ['TEST1'],
        'halogen': ['Cl'],
        'vmax': [20.0]
    })

    # Calculate charges first
    df_charged = calculator.batch_calculate_from_dataframe(df)

    # Test saving to CSV
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name

    try:
        calculator.save_charges(df_charged, temp_path)
        # Read back and check column name
        result_df = pd.read_csv(temp_path)
        assert 'dummy_charge_e' in result_df.columns
        assert len(result_df) == 1
        assert result_df.iloc[0]['dummy_charge_e'] != 0
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)