"""Tests for the Multiwfn parser module."""

import pytest
import tempfile
import os
from sigma_hole_docking.multiwfn_parser import MultiwfnParser


def test_surfanalysis_starred_max():
    """Synthetic _surfanalysis.txt with * row → correct Vmax."""
    parser = MultiwfnParser()

    # Create a synthetic _surfanalysis.txt content with a * row indicating the Vmax
    # Format: point_num a.u. eV kcal/mol X Y Z
    content = """Some header information
Number of surface maxima: 1
Number of surface minima: 0
# a.u. eV kcal/mol X/Y/Z coordinate(Angstrom)
    1    0.000  0.000    5.200    0.000    0.000    0.001
    2    0.000  0.000    6.300    0.500    0.000    0.001
*   3    0.000  0.000    7.100    1.000    0.000    0.001
    4    0.000  0.000    4.800    1.500    0.000    0.001
    5    0.000  0.000    3.900    2.000    0.000    0.001
===============================================================================
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='_surfanalysis.txt', delete=False) as f:
        f.write(content)
        file_path = f.name

    try:
        result = parser.parse_vmax_output(file_path)
        # Should find the value (7.100) from the line with * (the starred value is the Vmax)
        assert result['success'] == True
        assert abs(result['vmax'] - 7.100) < 0.001
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_au_conversion():
    """Value with a.u. → converted ×627.509."""
    parser = MultiwfnParser()

    # Create content with a.u. units that matches pattern 2 for a.u. detection
    content = """Some header information
The maximum value is 0.200 a.u.
Some more output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        file_path = f.name

    try:
        result = parser.parse_vmax_output(file_path)
        # Should find the value (0.200 a.u.) and convert to kcal/mol
        # 0.200 * 627.509 = 125.5018
        assert result['success'] == True
        assert abs(result['vmax'] - 125.5018) < 0.001
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_none_raw_match():
    """No pattern matched → no exception, success=False."""
    parser = MultiwfnParser()

    # Content that matches NO pattern
    content = """This file has no ESP data at all.
Just some random text.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        file_path = f.name

    try:
        result = parser.parse_vmax_output(file_path)
        assert result['success'] == False
        assert result['vmax'] is None
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_heuristic_fallback():
    """No standard pattern but a number in range → extracted with warning."""
    parser = MultiwfnParser()

    # Content with a number in the typical Vmax range but not in standard format
    content = """Some output
Maximum electrostatic potential: 25.5 kcal/mol
Some more output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        file_path = f.name

    try:
        result = parser.parse_vmax_output(file_path)
        # Should heuristically find 25.5
        assert result['success'] == True
        assert abs(result['vmax'] - 25.5) < 0.001
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)