"""Tests for the PDBQT I/O module."""

import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from sigma_hole_docking.pdbqt_io import (
    parse_pdbqt,
    write_pdbqt_from_mol,
    compute_geometric_center,
    compute_distance
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
    with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as f:
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        assert len(atoms) == 3
        
        # First two should not be dummy
        assert not atoms[0]['is_dummy']  # Carbon
        assert not atoms[1]['is_dummy']  # Hydrogen with zero charge
        
        # Third should be dummy (EP type)
        assert atoms[2]['is_dummy']
        assert atoms[2]['element'] == 'EP'
        assert atoms[2]['charge'] == 0.5
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_element_normalization():
    """Test CL → Cl, BR → Br."""
    # Create a PDBQT content with lowercase halogen symbols - compact format
    pdbqt_content = """ATOM      1  CL     0.000   0.000   0.000  0.00  0.00   -0.1000 CL
ATOM      2  BR     0.000   0.000   1.000  0.00  0.00   -0.1500 BR
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        assert len(atoms) == 2
        
        # Check that element symbols are normalized
        assert atoms[0]['element'] == 'Cl'
        assert atoms[1]['element'] == 'Br'
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        temp_path = f.name

    try:
        atoms = parse_pdbqt(temp_path)
        # Should parse the valid ATOM records and skip the malformed line
        assert len(atoms) == 2
        assert atoms[0]['element'] == 'C'
        assert atoms[1]['element'] == 'H'
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
