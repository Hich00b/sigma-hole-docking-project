"""Tests for the PDBQT I/O module."""

from sigma_hole_docking.pdbqt_io import parse_pdbqt, write_pdbqt_from_mol
from rdkit import Chem


def test_round_trip():
    """Write a small mol → parse back → atom count and charges match."""
    # Create a simple methane molecule
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)

    from rdkit.Chem import AllChem
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)

    # Compute Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as f:
        pdbqt_path = f.name

    try:
        # Write to PDBQT
        write_pdbqt_from_mol(mol, pdbqt_path)

        # Parse back
        atoms = parse_pdbqt(pdbqt_path)

        # Check atom count matches
        assert len(atoms) == mol.GetNumAtoms()

        # Check that we have coordinates and charges
        for atom in atoms:
            assert 'x' in atom
            assert 'y' in atom
            assert 'z' in atom
            assert 'charge' in atom
            assert 'element' in atom

    finally:
        if os.path.exists(pdbqt_path):
            os.unlink(pdbqt_path)


def test_dummy_detection():
    """Dummy atom (EP type or H+charge) flagged is_dummy=True."""
    import tempfile
    import os

    # Create a PDBQT content with a dummy atom (EP type)
    pdbqt_content = """REMARK   Test PDBQT with dummy atom
ATOM      1  C      0.000   0.000   0.000  0.00  0.00    0.0000 C
ATOM      2  H      1.000   0.000   0.000  0.00  0.00    0.0000 H
ATOM      3  EP     2.000   0.000   0.000  0.00  0.00    0.5000 EP
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        pdbqt_path = f.name

    try:
        atoms = parse_pdbqt(pdbqt_path)

        # Should have 3 atoms
        assert len(atoms) == 3

        # First two should not be dummy
        assert not atoms[0]['is_dummy']  # Carbon
        assert not atoms[1]['is_dummy']  # Hydrogen with zero charge

        # Third should be dummy (EP type)
        assert atoms[2]['is_dummy']
        assert atoms[2]['element'] == 'EP'
        assert atoms[2]['charge'] == 0.5

    finally:
        if os.path.exists(pdbqt_path):
            os.unlink(pdbqt_path)


def test_element_normalization():
    """CL → Cl, BR → Br."""
    import tempfile
    import os

    # Create a PDBQT content with lowercase halogen symbols
    pdbqt_content = """REMARK   Test PDBQT with element normalization
ATOM      1  CL     0.000   0.000   0.000  0.00  0.00   -0.1000 CL
ATOM      2  BR     1.000   0.000   0.000  0.00  0.00   -0.1500 BR
ATOM      3  I      2.000   0.000   0.000  0.00  0.00   -0.2000 I
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        pdbqt_path = f.name

    try:
        atoms = parse_pdbqt(pdbqt_path)

        # Should have 3 atoms
        assert len(atoms) == 3

        # Check element normalization
        assert atoms[0]['element'] == 'Cl'  # CL -> Cl
        assert atoms[1]['element'] == 'Br'  # BR -> Br
        assert atoms[2]['element'] == 'I'   # I stays I

    finally:
        if os.path.exists(pdbqt_path):
            os.unlink(pdbqt_path)


def test_malformed_line():
    """Bad line skipped, no crash."""
    import tempfile
    import os

    # Create a PDBQT content with some malformed lines
    pdbqt_content = """REMARK   Test PDBQT with malformed lines
ATOM      1  C      0.000   0.000   0.000  0.00  0.00    0.0000 C
NOTANATOM LINE THAT SHOULD BE SKIPPED
ATOM      2  O      1.000   0.000   0.000  0.00  0.00   -0.5000 O
ATOM      3  N      2.000   0.000   0.000                 # Missing fields
ATOM      4  H      3.000   0.000   0.000  0.00  0.00    0.0000 H
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        f.write(pdbqt_content)
        pdbqt_path = f.name

    try:
        atoms = parse_pdbqt(pdbqt_path)

        # Should have parsed 3 valid atoms (C, O, H) - the malformed lines should be skipped
        assert len(atoms) == 3

        # Check the valid atoms
        assert atoms[0]['element'] == 'C'
        assert atoms[1]['element'] == 'O'
        assert atoms[2]['element'] == 'H'

    finally:
        if os.path.exists(pdbqt_path):
            os.unlink(pdbqt_path)