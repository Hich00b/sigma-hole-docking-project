"""Tests for the charge calculator module."""

import os
import tempfile

import pandas as pd

from sigma_hole_docking.charge_calculator import SigmaHoleChargeCalculator, example_usage


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


def test_unknown_halogen_first_validation():
    """Test unknown halogen → raises ValueError on first validation (default_delta_r)."""
    calculator = SigmaHoleChargeCalculator()
    try:
        calculator.calculate_charge(10.0, "X")  # X not in default_delta_r
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown halogen: X" in str(e)
        assert "Supported:" in str(e)


def test_unknown_halogen_second_validation():
    """Test unknown halogen → raises ValueError on second validation (vdw_radii)."""
    calculator = SigmaHoleChargeCalculator()
    # Add a halogen to default_delta_r but not to vdw_radii
    calculator.default_delta_r["Y"] = 1.0  # Add fake halogen to default_delta_r
    # Don't add it to vdw_radii, so second validation will trigger
    try:
        calculator.calculate_charge(10.0, "Y")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown halogen: Y" in str(e)
        # This version doesn't include the "Supported:" message


def test_delta_r_exceeds_r_iso():
    """Test delta_r exceeds r_iso → gap clamped to 0.1 Å, warning logged."""
    calculator = SigmaHoleChargeCalculator()
    # For iodine: vdw_radii = 1.98 Å
    # Use delta_r = 2.0 Å (> 1.98) to trigger dist_gap <= 0
    result = calculator.calculate_charge(26.0, "I", delta_r=2.0)
    # Should not raise exception, just log warning and clamp
    assert isinstance(result, float)
    # With dist_gap clamped to 0.1, charge should be: 26.0 * 0.1 / 332.06
    expected_charge = 26.0 * 0.1 / 332.06
    assert abs(result - expected_charge) < 0.0001


def test_validation_error_warning():
    """Test charge validation warning when calculated Vmax differs significantly."""
    # Use charge_scale != 1.0 to trigger validation error
    calculator = SigmaHoleChargeCalculator(charge_scale=1.5)  # 50% scaling
    result = calculator.calculate_charge(26.0, "I")
    # With charge_scale=1.5, validation_error should be abs(1.5-1.0)*100 = 50% > 1.0%
    assert isinstance(result, float)
    # Expected charge: 26.0 * dist_gap / 332.06 * 1.5
    # For iodine: dist_gap = 1.98 - 1.2 = 0.78 (default values)
    expected_charge = 26.0 * 0.78 / 332.06 * 1.5
    assert abs(result - expected_charge) < 0.0001


def test_validation_error_pass():
    """Test charge validation passes when charge_scale is close to 1.0."""
    calculator = SigmaHoleChargeCalculator(charge_scale=1.005)  # 0.5% scaling
    result = calculator.calculate_charge(26.0, "I")
    # With charge_scale=1.005, validation_error should be abs(1.005-1.0)*100 = 0.5% < 1.0%
    assert isinstance(result, float)
    expected_charge = 26.0 * 0.78 / 332.06 * 1.005
    assert abs(result - expected_charge) < 0.0001


def test_batch_dataframe():
    """Test DataFrame input → dummy_charge column added."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({"halogen": ["I", "Br", "Cl"], "vmax": [20.0, 15.0, 10.0]})
    result_df = calculator.batch_calculate_from_dataframe(df)
    assert "dummy_charge" in result_df.columns
    assert len(result_df) == 3


def test_batch_dataframe_custom_delta_r():
    """Test DataFrame input with custom delta_r column."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame(
        {
            "halogen": ["I", "I"],
            "vmax": [20.0, 25.0],
            "custom_delta": [1.0, 1.5],  # Custom delta_r values
        }
    )
    result_df = calculator.batch_calculate_from_dataframe(df, delta_r_col="custom_delta")
    assert "dummy_charge" in result_df.columns
    assert len(result_df) == 2
    # Both should have calculated charges
    assert all(result_df["dummy_charge"].notna())


def test_save_charges_rename():
    """Test output CSV has dummy_charge column."""
    calculator = SigmaHoleChargeCalculator()
    df = pd.DataFrame({"halogen": ["I"], "vmax": [20.0]})

    # Test saving to CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name

    try:
        result_df = calculator.batch_calculate_from_dataframe(df)
        calculator.save_charges(result_df, temp_path)

        # Read back and check
        df_read = pd.read_csv(temp_path)
        assert "dummy_charge_e" in df_read.columns  # Renamed to dummy_charge_e
        assert len(df_read) == 1
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_example_usage():
    """Test example_usage function runs without error."""
    # This should not raise any exceptions
    result = example_usage()
    # The example_usage function returns the charge for iodobenzene
    assert isinstance(result, float)
    # Should be approximately the iodobenzene charge
    assert abs(result - 0.0611) < 0.001
