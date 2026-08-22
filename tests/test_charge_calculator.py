"""Tests for SigmaHoleChargeCalculator."""

import pandas as pd
import pytest

from sigma_hole_docking.charge_calculator import SigmaHoleChargeCalculator


class TestCalculateCharge:
    def test_iodine_typical_vmax(self):
        """iodobenzene: Vmax=26.0, I: r_iso=1.98, delta_r=1.2."""
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(26.0, "I")
        expected = 26.0 * (1.98 - 1.2) / 332.06
        assert abs(q - expected) < 1e-10
        assert abs(q - 0.0611) < 1e-4

    def test_chlorine(self):
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(20.0, "Cl")
        expected = 20.0 * (1.75 - 1.0) / 332.06
        assert abs(q - expected) < 1e-10

    def test_bromine(self):
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(22.0, "Br")
        expected = 22.0 * (1.83 - 1.1) / 332.06
        assert abs(q - expected) < 1e-10

    def test_fluorine(self):
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(15.0, "F")
        expected = 15.0 * (1.47 - 0.8) / 332.06
        assert abs(q - expected) < 1e-10

    def test_custom_delta_r(self):
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(26.0, "I", delta_r=1.5)
        expected = 26.0 * (1.98 - 1.5) / 332.06
        assert abs(q - expected) < 1e-10

    def test_zero_vmax(self):
        cc = SigmaHoleChargeCalculator()
        q = cc.calculate_charge(0.0, "I")
        assert q == 0.0

    def test_charge_scale(self):
        cc = SigmaHoleChargeCalculator(charge_scale=2.0)
        q = cc.calculate_charge(26.0, "I")
        expected = 2.0 * 26.0 * (1.98 - 1.2) / 332.06
        assert abs(q - expected) < 1e-10

    def test_invalid_halogen_raises(self):
        cc = SigmaHoleChargeCalculator()
        with pytest.raises((KeyError, ValueError)):
            cc.calculate_charge(26.0, "X")


class TestBatchCalculate:
    def test_batch_dataframe(self):
        """DataFrame in → dummy_charge column added."""
        cc = SigmaHoleChargeCalculator()
        df = pd.DataFrame(
            {
                "compound_id": ["c1", "c2"],
                "vmax": [26.0, 20.0],
                "halogen": ["I", "Cl"],
            }
        )
        result = cc.batch_calculate_from_dataframe(df)
        assert "dummy_charge" in result.columns
        assert len(result) == 2
        q1 = result.iloc[0]["dummy_charge"]
        assert abs(q1 - 26.0 * (1.98 - 1.2) / 332.06) < 1e-6


class TestSaveCharges:
    def test_save_charges_rename(self, tmp_path):
        """Output CSV has dummy_charge_e column."""
        cc = SigmaHoleChargeCalculator()
        df = pd.DataFrame(
            {
                "compound_id": ["c1"],
                "vmax": [26.0],
                "halogen": ["I"],
                "dummy_charge": [0.061],
            }
        )
        out = tmp_path / "charges.csv"
        cc.save_charges(df, str(out))
        loaded = pd.read_csv(out)
        assert "dummy_charge_e" in loaded.columns
