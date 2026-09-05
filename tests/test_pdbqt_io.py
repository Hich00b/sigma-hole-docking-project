"""Tests for the PDBQT I/O module."""

import os
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

from sigma_hole_docking.pdbqt_io import (
    compute_distance,
    compute_geometric_center,
    parse_pdbqt,
    parse_pdbqt_detailed,
    write_pdbqt_atoms,
    write_pdbqt_from_mol,
)

def test_round_trip():
    """Test write a small mol → parse back → atom count and charges match."""
    # Create a simple molecule (methane)
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)
    AllChem.ComputeGasteigerCharges(mol)

    # Test writing and reading back
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        temp_path = f.name

    try:
        success = write_pdbqt_from_mol(mol, temp_path)
        assert success

        # Parse back
        atoms = parse_pdbqt(temp_path)
        assert len(atoms) == 5  # C + 4H

        # Test compute_geometric_center
        center = compute_geometric_center(atoms)
        assert isinstance(center, tuple)
        assert len(center) == 3
        assert all(isinstance(c, float) for c in center)

        # Test compute_distance (distance from first to last atom should be > 0)
        if len(atoms) >= 2:
            dist = compute_distance(atoms[0], atoms[-1])
            assert isinstance(dist, float)
            assert dist >= 0.0

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_dummy_detection():
    """Test dummy atom (EP type or H+charge) flagged is_dummy=True."""
    # Create a PDBQT content with a dummy atom (EP type) - compact format
    pdbqt_content = """ATOM      1  C      0.000   0.000   0.000  0.00  0.00    0.0000 C
ATOM      2  H      0.000   0.000   1.000  0.00  0.00    0.0000 H
ATOM      3  EP     0.000   0.000   2.000  0.00  0.00    0.5000 EP
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        assert len(atoms) == 3

        # First two should not be dummy
        assert not atoms[0]["is_dummy"]  # Carbon
        assert not atoms[1]["is_dummy"]  # Hydrogen with zero charge

        # Third should be dummy (EP type)
        assert atoms[2]["is_dummy"]
        assert atoms[2]["element"] == "EP"
        assert atoms[2]["charge"] == 0.5
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_element_normalization():
    """Test CL → Cl, BR → Br."""
    # Create a PDBQT content with lowercase halogen symbols - compact format
    pdbqt_content = """ATOM      1  CL     0.000   0.000   0.000  0.00  0.00   -0.1000 CL
ATOM      2  BR     0.000   0.000   1.000  0.00  0.00   -0.1500 BR
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        assert len(atoms) == 2

        # Check that element symbols are normalized
        assert atoms[0]["element"] == "Cl"
        assert atoms[1]["element"] == "Br"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_malformed_line():
    """Test bad line skipped, no crash."""
    # Create a PDBQT content with some malformed lines - compact format
    pdbqt_content = """ATOM      1  C      0.000   0.000   0.000  0.00  0.00    0.0000 C
NOTANATOM line that should be skipped
ATOM      2  H      0.000   0.000   1.000  0.00  0.00    0.0000 H
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        # Should parse the valid ATOM records and skip the malformed line
        assert len(atoms) == 2
        assert atoms[0]["element"] == "C"
        assert atoms[1]["element"] == "H"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_parse_pdbqt_detailed():
    """Test parse_pdbqt_detailed function with extended format."""
    # Create a PDBQT content with extended format (with residue/chain fields)
    pdbqt_content = """ATOM      1  C      LIG A   1       0.000   0.000   0.000  0.00  0.0000  0.0000 C
ATOM      2  H      LIG A   1       0.000   0.000   1.000  0.00  0.0000  0.0000 H
ATOM      3  O      LIG A   1       0.000   0.000   2.000  0.00  -0.5000  0.0000 O
HETATM    4  EP     LIG A   1       0.000   0.000   3.000  0.00  0.3000  0.0000 EP
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt_detailed(temp_path)
        assert len(atoms) == 4

        # Check first atom (carbon)
        assert atoms[0]["index"] == 1
        assert atoms[0]["element"] == "C"
        assert atoms[0]["x"] == 0.0
        assert atoms[0]["y"] == 0.0
        assert atoms[0]["z"] == 0.0
        assert atoms[0]["charge"] == 0.0

        # Check second atom (hydrogen)
        assert atoms[1]["index"] == 2
        assert atoms[1]["element"] == "H"
        assert atoms[1]["x"] == 0.0
        assert atoms[1]["y"] == 0.0
        assert atoms[1]["z"] == 1.0
        assert atoms[1]["charge"] == 0.0

        # Check third atom (oxygen)
        assert atoms[2]["index"] == 3
        assert atoms[2]["element"] == "O"
        assert atoms[2]["x"] == 0.0
        assert atoms[2]["y"] == 0.0
        assert atoms[2]["z"] == 2.0
        assert atoms[2]["charge"] == -0.5

        # Check fourth atom (dummy EP)
        assert atoms[3]["index"] == 4
        assert atoms[3]["element"] == "EP"
        assert atoms[3]["x"] == 0.0
        assert atoms[3]["y"] == 0.0
        assert atoms[3]["z"] == 3.0
        assert atoms[3]["charge"] == 0.3
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_parse_pdbqt_detailed_with_invalid_lines():
    """Test parse_pdbqt_detailed skips invalid lines gracefully."""
    pdbqt_content = """ATOM      1  C      LIG A   1       0.000   0.000   0.000  0.00  0.0000  0.0000 C
ATOM      2  H      LIG A   1       0.000   0.000   not_a_number  0.00  0.0000  0.0000 H
ATOM      3  O      LIG A   1       0.000   0.000   2.000  0.00  -0.5000  0.0000 O
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt_detailed(temp_path)
        # Should parse 2 valid atoms, skip the invalid one
        assert len(atoms) == 2
        assert atoms[0]["element"] == "C"
        assert atoms[1]["element"] == "O"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
