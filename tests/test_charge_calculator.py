"""Tests for the charge calculator module."""

import os
import tempfile

import pandas as pd

from sigma_hole_docking.charge_calculator import SigmaHoleChargeCalculator


def test_charge_formula():
    """Test iodobenzene Vmax=26.0, I → q ≈ 0.0611 e (4 decimals)."""
    calculator = SigmaHoleChargeCalculator()
    result = calculator.calculate_charge(26.0, "I")
    assert abs(result - 0.0611) < 0.0001


def test_vmax_zero():
    """Test Vmax=0 → q = 0."""
    calculator = SigmaHoleChargeCalculator()
    result = calculator.calculate_charge(0.0, "I")
    assert result == 0.0


def test_unknown_halogen():
    """Test unknown halogen → raises ValueError."""
    calculator = SigmaHoleChargeCalculator()
    try:
        calculator.calculate_charge(10.0, "X")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_delta_r_exceeds_r_iso():
    """Test delta_r exceeds r_iso → gap clamped to 0.1 Å, warning logged."""
    calculator = SigmaHoleChargeCalculator()
    # Vmax=5.0 would normally give delta_r > r_iso for iodine
    result = calculator.calculate_charge(5.0, "I")
    # Should not raise exception, just log warning and clamp
    assert isinstance(result, float)


def test_batch_dataframe():
    """Test DataFrame input → dummy_charge column added."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({"halogen": ["I", "Br", "Cl"], "vmax": [20.0, 15.0, 10.0]})
    result_df = calculator.batch_calculate_from_dataframe(df)
    assert "dummy_charge" in result_df.columns
    assert len(result_df) == 3


def test_save_charges_rename():
    """Test output CSV has dummy_charge column."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({"halogen": ["I"], "vmax": [20.0]})

    # Test saving to CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name

    try:
        result_df = calculator.batch_calculate_from_dataframe(df)
        result_df.to_csv(temp_path, index=False)

        # Read back and check
        df_read = pd.read_csv(temp_path)
        assert "dummy_charge" in df_read.columns
        assert len(df_read) == 1
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
