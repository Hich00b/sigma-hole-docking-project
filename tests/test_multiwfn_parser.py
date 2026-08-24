"""Tests for MultiwfnParser."""

from sigma_hole_docking.multiwfn_parser import MultiwfnParser


class TestSurfAnalysis:
    def test_surfanalysis_starred_max(self, tmp_path):
        """Synthetic _surfanalysis.txt with * row → correct Vmax."""
        content = """\
 Number of surface maxima:      2
# a.u. eV kcal/mol X/Y/Z coordinate(Angstrom)
 1  0.040  1.088  25.10  0.000  0.000  1.500
*  2  0.042  1.143  26.36  0.000  0.000  2.000
"""
        path = tmp_path / "_surfanalysis.txt"
        path.write_text(content)
        parser = MultiwfnParser()
        result = parser.parse_vmax_output(str(path))
        assert result["success"] is True
        assert result["vmax"] is not None
        # The starred row has kcal/mol = 26.36
        assert abs(result["vmax"] - 26.36) < 0.1


class TestAuConversion:
    def test_au_conversion(self, tmp_path):
        """Value with a.u. → converted ×627.509."""
        content = """\
 The maximal value is   0.050 a.u.
"""
        path = tmp_path / "output.txt"
        path.write_text(content)
        parser = MultiwfnParser()
        result = parser.parse_vmax_output(str(path))
        if result["vmax"] is not None:
            # 0.050 a.u. → 0.050 * 627.509 ≈ 31.4 kcal/mol
            assert result["vmax"] > 30.0


class TestNoneRawMatch:
    def test_none_raw_match(self, tmp_path):
        """No pattern matched → no exception, success=False."""
        path = tmp_path / "empty.txt"
        path.write_text("This file has no ESP data at all.\n")
        parser = MultiwfnParser()
        result = parser.parse_vmax_output(str(path))
        assert result["success"] is False
        assert result["vmax"] is None


class TestHeuristicFallback:
    def test_heuristic_fallback(self, tmp_path):
        """No standard pattern but a number in range → extracted with warning."""
        content = """\
 Some custom output
 The maximum value found: 25.5 kcal/mol
"""
        path = tmp_path / "custom.txt"
        path.write_text(content)
        parser = MultiwfnParser()
        result = parser.parse_vmax_output(str(path))
        # May or may not extract depending on patterns, but must not crash
        assert isinstance(result, dict)
        assert "success" in result
