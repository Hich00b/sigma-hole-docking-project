"""
Sigma Hole Receptor Processor

Prepares receptor PDBQT files for sigma-hole docking studies.
Handles protein/ligand receptor preparation, charge assignment, and PDBQT formatting.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from . import pdbqt_io

logger = logging.getLogger(__name__)


class SigmaHoleReceptorProcessor:
    """
    Processes receptor files for sigma-hole docking.
    """

    # Electronegativity-based fallback partial charges (Pauling scale)
    # Used when Gasteiger returns NaN (common for iodine-containing molecules)
    _fallback_charges: dict[str, float] = {
        "H": 0.05,
        "C": -0.10,
        "N": -0.30,
        "O": -0.40,
        "S": -0.15,
        "F": -0.25,
        "Cl": -0.15,
        "Br": -0.10,
        "I": -0.05,
        "At": 0.0,
    }

    def __init__(self):
        """Initialize the receptor processor."""

    def _fix_nan_charges(self, mol: Chem.Mol) -> None:
        """Replace NaN Gasteiger charges with electronegativity-based fallback values.

        RDKit's ComputeGasteigerCharges returns NaN for atoms it can't parameterize
        (especially iodine). This method fills in reasonable fallback charges so the
        PDBQT output has no zero-charge atoms.
        """
        import math

        fixed = 0
        for atom in mol.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                charge = atom.GetDoubleProp("_GasteigerCharge")
                if math.isnan(charge) or math.isinf(charge):
                    fallback = self._fallback_charges.get(atom.GetSymbol(), 0.0)
                    atom.SetDoubleProp("_GasteigerCharge", fallback)
                    fixed += 1
        if fixed > 0:
            logger.info(f"Fixed {fixed} NaN/Inf Gasteiger charges with fallback values")

    def prepare_receptor_from_pdb(
        self, pdb_path: str, output_path: str, add_hydrogens: bool = True, optimize: bool = True
    ) -> bool:
        """
        Prepare receptor PDBQT from PDB file.

        Args:
            pdb_path: Path to input PDB file
            output_path: Path to save PDBQT file
            add_hydrogens: Whether to add hydrogens (kept for compatibility but not used)
            optimize: Whether to optimize geometry with force field

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read PDB file - support both ATOM and HETATM records (for OpenBabel compatibility)
            mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
            if mol is None:
                logger.error(
                    f"Failed to read PDB file: {pdb_path}. Please ensure the file is in valid PDB format with proper ATOM or HETATM records. Common issues: missing END record, incorrect formatting, or unsupported elements."
                )
                return False

            # We keep existing hydrogens from the PDB; do not add extra ones
            # to avoid incorrect hydrogen addition due to poor bond perception.
            if optimize and mol.GetNumConformers() == 0:
                # Generate 3D coordinates if needed
                if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                    logger.warning("Failed to generate 3D coordinates, using existing")
                else:
                    # Optimize with MMFF
                    AllChem.MMFFOptimizeMolecule(mol)

            # Compute Gasteiger charges for better electrostatics
            try:
                AllChem.ComputeGasteigerCharges(mol)
                self._fix_nan_charges(mol)
            except Exception as e:
                logger.warning(f"Failed to compute Gasteiger charges: {e}, using fallback values")
                self._fix_nan_charges(mol)  # This will set all atoms to fallback charges

            # Create PDBQT manually for better control over atom types and charges
            self._create_pdbqt_manual(mol, output_path)

            logger.info(f"Generated receptor PDBQT manually: {output_path}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                f"Error preparing receptor from PDB {pdb_path}: {e}. Check PDB file format and try again."
            )
            return False

    def prepare_receptor_from_smiles(
        self, smiles: str, output_path: str, add_hydrogens: bool = True, optimize: bool = True
    ) -> bool:
        """
        Prepare receptor PDBQT from SMILES string.

        Args:
            smiles: SMILES string of the receptor/ligand
            output_path: Path to save PDBQT file
            add_hydrogens: Whether to add hydrogens
            optimize: Whether to optimize geometry with force field

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Failed to parse SMILES: {smiles}")
                return False

            if add_hydrogens:
                mol = Chem.AddHs(mol, addCoords=True)

            if optimize and mol.GetNumConformers() == 0:
                # Generate 3D coordinates
                if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                    logger.error(f"Failed to generate 3D coordinates for {smiles}")
                    return False
                # Optimize with MMFF
                AllChem.MMFFOptimizeMolecule(mol)
                # Compute Gasteiger charges
                AllChem.ComputeGasteigerCharges(mol)
                self._fix_nan_charges(mol)

            # Create PDBQT manually for better control
            self._create_pdbqt_manual(mol, output_path)

            logger.info(f"Generated receptor PDBQT from SMILES: {output_path}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error preparing receptor from SMILES: {e}")
            return False

    def _create_pdbqt_manual(self, mol: Chem.Mol, output_path: str) -> None:
        """
        Manually create PDBQT file using the shared PDBQT I/O module.

        Args:
            mol: RDKit molecule
            output_path: Path to save PDBQT file
        """

        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        # Prepare atoms list for the shared I/O function
        atoms = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)

            element = atom.GetSymbol()
            # Get Gasteiger charge if available
            try:
                charge = atom.GetDoubleProp("_GasteigerCharge")
            except Exception as e:
                logger.debug(
                    f"Could not get Gasteiger charge for atom {atom.GetIdx()}: {e}, defaulting to 0.0"
                )
                charge = 0.0

            # Simple atom type mapping (can be improved)
            atom_type = element
            if element == "C":
                # Distinguish between different carbon types if needed
                atom_type = "C"
            elif element == "O":
                atom_type = "OA"  # Carbonyl oxygen
            elif element == "N":
                atom_type = "N"  # or 'NA', 'NC', etc. depending on context
            elif element == "S":
                atom_type = "S"

            atoms.append(
                {
                    "element": element,
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z,
                    "charge": charge,
                    "atom_type": atom_type,
                }
            )

        # Use shared function to write the basic PDBQT format
        title = "Generated by Sigma Hole Receptor Processor"
        success = pdbqt_io.write_pdbqt_atoms(
            atoms=atoms,
            output_path=output_path,
            title=title,
            is_docking=True,  # Include ROOT/ENDROOT/TORSDOF sections
        )

        if not success:
            logger.error(f"Failed to create PDBQT file: {output_path}")
        else:
            # Add rotatable bonds count to TORSDOF section
            try:
                from rdkit.Chem import AllChem

                rotatable_bonds = AllChem.CalcNumRotatableBonds(mol)
                with open(output_path, "r+") as f:
                    content = f.read()
                    # Find and replace the rotatable_bonds line (assumes it's the last line)
                    lines = content.split("\n")
                    if lines and lines[-1].strip() == "0":
                        lines[-1] = str(rotatable_bonds)
                        f.seek(0)
                        f.write("\n".join(lines))
                        f.truncate()
            except Exception as e:
                logger.debug(f"Could not compute rotatable bonds: {e}, leaving as 0")
                # If we can't compute rotatable bonds, just leave it as 0

        logger.debug(f"Created manual PDBQT with {num_atoms} atoms")

    def prepare_acetone_receptor(self, output_path: str = "receptor.pdbqt") -> bool:
        """
        Prepare a simple acetone receptor for testing/sigma-hole validation.

        This creates acetone from SMILES CC(=O)C, adds hydrogens,
        optimizes geometry, computes Gasteiger charges, and writes a proper PDBQT.

        Args:
            output_path: Path to save the receptor PDBQT file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use SMILES to generate acetone, then process
            return self.prepare_receptor_from_smiles(
                "CC(=O)C", output_path, add_hydrogens=True, optimize=True
            )
        except Exception as e:
            logger.error(f"Error creating acetone receptor: {e}")
            return False

    def batch_prepare_receptors(
        self,
        receptor_df: pd.DataFrame,
        output_dir: str,
        input_col: str = "pdb_path",
        id_col: str = "receptor_id",
        input_type: str = "pdb",
    ) -> list[str]:
        """
        Prepare receptor PDBQT files for a batch of receptors.

        Args:
            receptor_df: DataFrame with receptor information
            output_dir: Directory to save PDBQT files
            input_col: Column name for input file paths or SMILES
            id_col: Column name for receptor identifiers
            input_type: Type of input ('pdb' or 'smiles')

        Returns:
            List of generated receptor PDBQT file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        for _, row in receptor_df.iterrows():
            try:
                receptor_id = row[id_col]
                input_data = row[input_col]
                output_path = os.path.join(output_dir, f"{receptor_id}_receptor.pdbqt")

                if input_type == "pdb":
                    success = self.prepare_receptor_from_pdb(input_data, output_path)
                elif input_type == "smiles":
                    success = self.prepare_receptor_from_smiles(input_data, output_path)
                else:
                    logger.error(f"Unsupported input_type: {input_type}")
                    success = False

                if success:
                    generated_files.append(output_path)
                    logger.info(f"Generated receptor for {receptor_id}")
                else:
                    logger.error(f"Failed to generate receptor for {receptor_id}")

            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Error processing receptor {row.get(id_col, 'unknown')}: {e}")

        logger.info(f"Generated {len(generated_files)} receptor PDBQT files")
        return generated_files


def example_usage():
    """Example usage of the receptor processor."""
    processor = SigmaHoleReceptorProcessor()

    # Example 1: Create acetone receptor (standard test case)
    print("Creating acetone receptor...")
    success = processor.prepare_acetone_receptor("acetone_receptor.pdbqt")
    if success:
        print("Successfully created acetone_receptor.pdbqt")
        # Show the file
        try:
            with open("acetone_receptor.pdbqt", "r") as f:
                content = f.read()
                print("Acetone receptor PDBQT:")
                print(content)
        except Exception as e:
            print(f"Could not read receptor file: {e}")
    else:
        print("Failed to create acetone receptor")

    # Example 2: Create receptor from SMILES (e.g., simple ketone)
    print("\nCreating receptor from SMILES (acetone)...")
    success = processor.prepare_receptor_from_smiles("CC(=O)C", "acetone_from_smiles.pdbqt")
    if success:
        print("Successfully created acetone_from_smiles.pdbqt")
    else:
        print("Failed to create receptor from SMILES")

    return success


if __name__ == "__main__":
    example_usage()
