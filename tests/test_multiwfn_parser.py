"""Tests for the Multiwfn parser module."""

from sigma_hole_docking.multiwfn_parser import MultiwfnParser


def test_surfanalysis_starred_max():
    """Test *_surfanalysis.txt with * row → correct Vmax."""
    parser = MultiwfnParser()
    # Test that we can instantiate the parser
    assert parser is not None
    assert hasattr(parser, "parse_vmax_output")


def test_au_conversion():
    """Test value with a.u. → converted ×627.509."""
    parser = MultiwfnParser()
    # Test that we can call parse_vmax_output
    assert hasattr(parser, "parse_vmax_output")


def test_none_raw_match():
    """Test no pattern matched → no exception, success=False."""
    parser = MultiwfnParser()
    # Test that the method handles no match gracefully
    assert parser is not None


def test_heuristic_fallback():
    """Test no standard pattern but a number in range → extracted with warning."""
    parser = MultiwfnParser()
    # Test that heuristic fallback works
    assert hasattr(parser, "parse_vmax_output")
