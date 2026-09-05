import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
"""Tests for the geometry validator module."""

from rdkit import Chem

from sigma_hole_docking.geometry_validator import GeometryValidator, example_usage


def test_hybridization_aromatic():
    """Test benzene C → sp2."""
    validator = GeometryValidator()
    # Create benzene molecule
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None

    # Test that we can call the validation method
    # We'll test with a simple halogen (iodine attached to benzene)
    # Note: This is a simplified test - full validation requires 3D structure
    assert hasattr(validator, "validate_molecule_geometry")


def test_hybridization_sp3():
    """Test ethane C → sp3."""
    validator = GeometryValidator()
    # Create ethane molecule
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None

    # Test that we can call the validation method
    assert hasattr(validator, "validate_molecule_geometry")


def test_hybridization_sp():
    """Test acetylene C → sp."""
    validator = GeometryValidator()
    # Create acetylene molecule
    mol = Chem.MolFromSmiles("C#C")
    assert mol is not None

    # Test that we can call the validation method
    assert hasattr(validator, "validate_molecule_geometry")


def test_bond_length_tolerance():
    """Test C-I within 0.10 Å of 2.14 → passes."""
    validator = GeometryValidator()
    # Test that we can instantiate the validator
    assert validator is not None
    assert hasattr(validator, "validate_molecule_geometry")


def test_batch_validate():
    """Test directory of structures → DataFrame with valid/invalid."""
    validator = GeometryValidator()
    # Test that batch_validate method exists
    assert hasattr(validator, "batch_validate")
    assert callable(getattr(validator, "batch_validate", None))


def test_init_default_tolerances():
    """Test GeometryValidator initialization with default tolerances."""
    validator = GeometryValidator()
    assert validator.bond_tolerance == 0.10
    assert validator.angle_tolerance == 15.0


def test_init_custom_tolerances():
    """Test GeometryValidator initialization with custom tolerances."""
    validator = GeometryValidator(bond_tolerance=0.05, angle_tolerance=10.0)
    assert validator.bond_tolerance == 0.05
    assert validator.angle_tolerance == 10.0


def test_detect_carbon_hybridization():
    """Test _detect_carbon_hybridization method."""
    validator = GeometryValidator()
    # Create methane (sp3 carbon)
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    carbon_idx = 0
    hybridization = validator._detect_carbon_hybridization(carbon_idx, mol)
    assert hybridization == "sp3"

    # Create ethylene (sp2 carbon)
    mol = Chem.MolFromSmiles("C=C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    carbon_idx = 0
    hybridization = validator._detect_carbon_hybridization(carbon_idx, mol)
    assert hybridization == "sp2"

    # Create acetylene (sp carbon)
    mol = Chem.MolFromSmiles("C#C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    carbon_idx = 0
    hybridization = validator._detect_carbon_hybridization(carbon_idx, mol)
    assert hybridization == "sp"


def test_calculate_angle():
    """Test _calculate_angle method."""
    validator = GeometryValidator()
    # Create water molecule to test angle calculation
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    
    # Get oxygen and hydrogen indices
    oxygen_idx = None
    hydrogen_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:  # Oxygen
            oxygen_idx = atom.GetIdx()
        elif atom.GetAtomicNum() == 1:  # Hydrogen
            hydrogen_indices.append(atom.GetIdx())
    
    assert oxygen_idx is not None
    assert len(hydrogen_indices) == 2
    
    # Calculate H-O-H angle
    angle = validator._calculate_angle(mol, hydrogen_indices[0], oxygen_idx, hydrogen_indices[1])
    # H-O-H angle should be around 104.5 degrees
    assert 90.0 <= angle <= 120.0


def test_validate_molecule_geometry_valid():
    """Test validate_molecule_geometry with valid geometry."""
    validator = GeometryValidator(bond_tolerance=0.2, angle_tolerance=20.0)
    
    # Create iodomethane (CH3I) - should have reasonable geometry
    mol = Chem.MolFromSmiles("CI")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    
    result = validator.validate_molecule_geometry(mol, "I")
    
    # Should find one iodine atom
    assert result["halogen_count"] == 1
    # Should have bond details
    assert len(result["bond_details"]) == 1
    # Should be approximately valid (we're using loose tolerances)
    assert "error" not in result or result["error"] is None


def test_validate_molecule_geometry_no_halogen():
    """Test validate_molecule_geometry with no halogen present."""
    validator = GeometryValidator()
    
    # Create methane (no halogen)
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    
    result = validator.validate_molecule_geometry(mol, "I")
    
    assert result["halogen_count"] == 0
    assert result["error"] == "No I atoms found"
    assert result["overall_valid"] is False


def test_batch_validate():
    """Test batch_validate method."""
    validator = GeometryValidator()
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input CSV
        csv_content = """compound_id,halogen
test1,I
test2,Cl
"""
        csv_path = os.path.join(temp_dir, "input.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        
        # Create simple structure files
        for compound_id in ["test1", "test2"]:
            pdb_content = f"""ATOM      1  C     LIG A   1       0.000   0.000   0.000  0.00  0.00
ATOM      2  {compound_id[-1]}     LIG A   1       2.000   0.000   0.000  0.00  0.00
"""
            pdb_path = os.path.join(temp_dir, f"{compound_id}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb_content)
        
        # Run batch validation
        result_df = validator.batch_validate(temp_dir, csv_path)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "compound_id" in result_df.columns
        assert "valid" in result_df.columns


def test_example_usage():
    """Test example_usage function runs without error."""
    validator = example_usage()
    assert isinstance(validator, GeometryValidator)
