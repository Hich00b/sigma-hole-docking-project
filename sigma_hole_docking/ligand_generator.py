"""
Sigma Hole Ligand Generator

Creates PDBQT files with dummy atoms positioned for sigma-hole modeling.
Handles ligand preparation, dummy atom placement, and PDBQT formatting.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from . import pdbqt_io

logger = logging.getLogger(__name__)


def preprocess_smiles(smiles: str) -> str:
    """
    Preprocess a SMILES string to handle common parsing issues.
    Tries to generate a kekulized version if the original fails.
    """
    from rdkit import Chem

    # First try to parse normally
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.Kekulize(mol)
            return Chem.MolToSmiles(mol)
        except (ValueError, RuntimeError):
            # If kekulization fails, continue with non-kekulized mol but try fixes first
            fixed_smiles = _fix_known_problematic_smiles(smiles)
            if fixed_smiles != smiles:
                # Try parsing the fixed version
                fixed_mol = Chem.MolFromSmiles(fixed_smiles)
                if fixed_mol is not None:
                    try:
                        Chem.Kekulize(fixed_mol)
                        return Chem.MolToSmiles(fixed_mol)
                    except (ValueError, RuntimeError):
                        # If fixed version still won't kekulize, use it as-is
                        mol = fixed_mol
            # If no fix or fix didn't help, continue with original non-kekulized mol
    else:
        # If normal parsing fails completely, try to fix known problematic patterns
        fixed_smiles = _fix_known_problematic_smiles(smiles)
        if fixed_smiles != smiles:
            # Try parsing the fixed version
            fixed_mol = Chem.MolFromSmiles(fixed_smiles, sanitize=False)
            if fixed_mol is not None:
                # Try to kekulize the fixed version
                try:
                    Chem.Kekulize(fixed_mol)
                    return Chem.MolToSmiles(fixed_mol)
                except (ValueError, RuntimeError):
                    # If still can't kekulize, continue processing the fixed version
                    mol = fixed_mol
            # If fixed version can't be parsed, fall back to original approach below
        # If no fix or fix didn't work with sanitize=False, continue with original approach

    # If we couldn't parse at all, or if kekulization failed and fixes didn't work, try without kekulization initially
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return smiles  # Return original if we can't parse at all

    # Try to sanitize without kekulization
    try:
        sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Could not sanitize mol: {e}, continuing anyway")

    # Add hydrogens
    try:
        mol = Chem.AddHs(mol)
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Could not add hydrogens: {e}, continuing anyway")

    # Try to generate SMILES
    try:
        return Chem.MolToSmiles(mol)
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Could not generate SMILES: {e}, returning original")
        # If we can't generate SMILES, return original
        return smiles


def _fix_known_problematic_smiles(smiles: str) -> str:
    """
    Apply fixes for known problematic SMILES patterns.
    """
    # Handle 2-chloropyrimidine and similar problematic heterocycles
    if smiles == "n1ccnc1Cl":
        # Convert to equivalent but more parsable form: c1cnc(Cl)nc1
        # This places the chlorine correctly bonded to a carbon in the pyrimidine ring
        return "c1cnc(Cl)nc1"

    # Add other known problematic patterns here as needed
    # For example, other chloropyrimidines:
    elif smiles == "n1ccc(nc1)Cl":  # 4-chloropyrimidine
        return "c1cncc(Cl)n1"
    elif (
        smiles == "n1ccn(c1)Cl"
    ):  # Already fixed form for 2-chloropyrimidine (though wrong bonding)
        return "c1cnc(Cl)nc1"

    # Return original if no fix applies
    return smiles


class SigmaHoleLigandGenerator:
    """
    Generates ligand PDBQT files with dummy atoms for sigma-hole interactions.
    """

    def __init__(self):
        """Initialize the ligand generator."""
        # Default distances for halogen dummy atom placement
        # These are typical values: distance from halogen nucleus to dummy atom
        self.default_delta_r = {
            "F": 0.8,  # Fluorine
            "Cl": 1.0,  # Chlorine
            "Br": 1.1,  # Bromine
            "I": 1.2,  # Iodine
            "At": 1.3,  # Astatine
        }

        # Electronegativity-based fallback partial charges (Pauling scale)
        # Used when Gasteiger returns NaN (common for iodine-containing molecules)
        self._fallback_charges = {
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

    def _get_halogen_position(
        self, mol: Chem.Mol, halogen_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the position of a halogen atom and its bonded carbon.

        Args:
            mol: RDKit molecule object
            halogen_idx: Index of the halogen atom

        Returns:
            Tuple of (halogen_position, carbon_position) as numpy arrays
        """
        halogen_atom = mol.GetAtomWithIdx(halogen_idx)
        halogen_pos = np.array(mol.GetConformer().GetAtomPosition(halogen_idx))

        # Find bonded carbon (should be exactly one for halogen in aromatic/alkyl halide)
        carbon_neighbors = [
            nbr.GetIdx()
            for nbr in halogen_atom.GetNeighbors()
            if nbr.GetAtomicNum() == 6  # Carbon
        ]

        if not carbon_neighbors:
            raise ValueError(f"No carbon neighbor found for halogen at index {halogen_idx}")

        carbon_idx = carbon_neighbors[0]
        carbon_pos = np.array(mol.GetConformer().GetAtomPosition(carbon_idx))

        return halogen_pos, carbon_pos

    def _place_dummy_atom(
        self, halogen_pos: np.ndarray, carbon_pos: np.ndarray, distance: float = 1.2
    ) -> np.ndarray:
        """
        Place dummy atom along the C-X bond axis, extending beyond the halogen.

        Args:
            halogen_pos: Position of halogen atom [x, y, z]
            carbon_pos: Position of bonded carbon atom [x, y, z]
            distance: Distance from halogen to dummy atom (A)

        Returns:
            Position of dummy atom as numpy array [x, y, z]
        """
        # Vector from carbon to halogen (points from C to X)
        c_to_x = halogen_pos - carbon_pos
        c_to_x_unit = c_to_x / np.linalg.norm(c_to_x)

        # Dummy atom is placed along this vector, beyond the halogen
        # Position = halogen_position + (distance * unit_vector_from_C_to_X)
        dummy_pos = halogen_pos + (distance * c_to_x_unit)

        return dummy_pos

    def prepare_ligand_from_smiles(
        self,
        smiles: str,
        halogen: str,
        charge: float,
        output_path: str,
        delta_r: float = 1.2,
        add_dummy: bool = True,
    ) -> bool:
        """
        Prepare ligand PDBQT from SMILES with optional dummy atom for sigma-hole.

        Args:
            smiles: SMILES string of the ligand
            halogen: Halogen symbol ('F', 'Cl', 'Br', 'I')
            charge: Charge for the dummy atom (electron units)
            output_path: Path to save PDBQT file
            delta_r: Distance from halogen to dummy atom (A)
            add_dummy: Whether to add dummy atom (True for sigma-hole, False for control)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Preprocess SMILES to handle kekulization issues
            processed_smiles = preprocess_smiles(smiles)
            if processed_smiles != smiles:
                logger.debug(f"Preprocessed SMILES: {smiles} -> {processed_smiles}")

            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(processed_smiles)
            if mol is None:
                logger.error(f"Failed to parse SMILES: {smiles}")
                return False

            # Add hydrogens for proper geometry
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                logger.error(f"Failed to generate 3D coordinates for {smiles}")
                return False

            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)

            # Compute Gasteiger partial charges BEFORE any atom manipulation.
            # Without this call, HasProp('_GasteigerCharge') always returns False -> charge = 0.0
            # This was the root cause of all-zero charges in PDBQT output.
            AllChem.ComputeGasteigerCharges(mol)
            self._fix_nan_charges(mol)
            logger.info(f"Computed Gasteiger charges for {smiles}")

            # Find ALL halogen atoms of the specified type (supports multi-halogen molecules)
            halogen_indices = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == halogen:
                    halogen_indices.append(atom.GetIdx())

            if not halogen_indices:
                logger.error(f"No {halogen} atom found in {smiles}")
                return False

            # Create molecule for PDBQT generation
            mol_with_dummy = Chem.RWMol(mol)
            dummy_indices = []

            # Add dummy atoms for ALL halogens (not just the first one)
            if add_dummy:
                # Distribute total dummy charge equally among all halogens of same type
                per_halogen_charge = charge / len(halogen_indices)

                for halogen_idx in halogen_indices:
                    halogen_pos, carbon_pos = self._get_halogen_position(mol, halogen_idx)
                    dummy_pos = self._place_dummy_atom(halogen_pos, carbon_pos, delta_r)

                    # Get the original halogen charge for charge conservation
                    halogen_atom = mol_with_dummy.GetAtomWithIdx(halogen_idx)
                    original_halogen_charge = (
                        halogen_atom.GetDoubleProp("_GasteigerCharge")
                        if halogen_atom.HasProp("_GasteigerCharge")
                        else 0.0
                    )

                    # Add dummy atom (as hydrogen with custom charge)
                    dummy_idx = mol_with_dummy.AddAtom(Chem.Atom(1))  # Hydrogen
                    mol_with_dummy.GetAtomWithIdx(dummy_idx).SetDoubleProp(
                        "dummy_charge", per_halogen_charge
                    )
                    dummy_indices.append(dummy_idx)

                    # Charge conservation: dummy carries +q, so halogen must lose q to keep net charge zero.
                    # q_hal_new = q_hal_original - q_dummy
                    new_halogen_charge = original_halogen_charge - per_halogen_charge
                    halogen_atom.SetDoubleProp("_GasteigerCharge", new_halogen_charge)

                    # Set position of dummy atom
                    conf = mol_with_dummy.GetConformer()
                    conf.SetAtomPosition(dummy_idx, dummy_pos.tolist())

                logger.info(
                    f"Added {len(dummy_indices)} dummy atom(s) for {len(halogen_indices)} {halogen} atom(s)"
                )
            else:
                dummy_indices = None  # No dummy atoms

            # Use manual PDBQT creation (more reliable than OpenBabel)
            self._create_pdbqt_manual(mol_with_dummy, dummy_indices, charge, output_path, add_dummy)
            logger.info(f"Generated PDBQT manually: {output_path}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error preparing ligand from SMILES: {e}")
            return False

    def _create_pdbqt_manual(
        self,
        mol: Chem.RWMol,
        dummy_indices: list[int] | None,
        charge: float,
        output_path: str,
        add_dummy: bool = True,
    ) -> None:
        """
        Manually create PDBQT file using the shared PDBQT I/O module.

        Args:
            mol: RDKit molecule (with or without dummy atoms)
            dummy_indices: List of dummy atom indices (None if no dummy atoms)
            charge: Total charge for all dummy atoms combined
            output_path: Path to save PDBQT file
            add_dummy: Whether dummy atoms are present
        """
        # Prepare title based on whether dummy atoms are present
        if add_dummy:
            title = "Generated by Sigma Hole Ligand Generator"
        else:
            title = "Generated by Sigma Hole Ligand Generator (Control - No Dummy)"

        # Use the shared PDBQT I/O function
        success = pdbqt_io.write_pdbqt_from_mol(
            mol=mol,
            output_path=output_path,
            title=title,
            is_docking=True,
            dummy_indices=dummy_indices,
            total_dummy_charge=charge if add_dummy else 0.0,
        )

        if not success:
            logger.error(f"Failed to create PDBQT file: {output_path}")
        else:
            if add_dummy:
                logger.info(
                    f"Created manual PDBQT with {len(set(dummy_indices) if dummy_indices else set())} dummy atom(s): {output_path}"
                )
            else:
                logger.info(f"Created manual PDBQT (control): {output_path}")

    def prepare_ligand_from_structure(
        self,
        structure_path: str,
        halogen: str,
        charge: float,
        output_path: str,
        delta_r: float = 1.2,
        add_dummy: bool = True,
        structure_format: str = "auto",
    ) -> bool:
        """
        Prepare ligand PDBQT from an existing structure file (PDB, SDF, MOL2).
        No geometry optimization is performed — input coordinates are preserved,
        which is essential for DFT-optimized structures where Vmax was measured.

        Args:
            structure_path: Path to input structure file
            halogen: Halogen symbol ('F', 'Cl', 'Br', 'I')
            charge: Charge for the dummy atom (electron units)
            output_path: Path to save PDBQT file
            delta_r: Distance from halogen to dummy atom (A)
            add_dummy: Whether to add dummy atom (True for sigma-hole, False for control)
            structure_format: File format — 'pdb', 'sdf', 'mol2', or 'auto' (detect from extension)

        Returns:
            True if successful, False otherwise
        """
        print("=" * 60)
        print("!!! CONFIRMED: YOUR UPDATED LIGAND GENERATOR IS RUNNING !!!")
        print(f"!!! structure_path: {structure_path}")
        print(f"!!! halogen: {halogen}, charge: {charge}, add_dummy: {add_dummy}")
        print("=" * 60 + "\n")

        try:
            # Auto-detect format from extension
            if structure_format == "auto":
                ext = os.path.splitext(structure_path)[1].lower()
                format_map = {".pdb": "pdb", ".sdf": "sdf", ".mol": "sdf", ".mol2": "mol2"}
                structure_format = format_map.get(ext, "")
                if not structure_format:
                    logger.error(f"Unsupported structure format: {ext}")
                    return False

            # Read structure file (no geometry optimization, no H addition)
            if structure_format == "pdb":
                try:
                    mol = Chem.MolFromPDBFile(structure_path, removeHs=False)
                except OSError:
                    mol = None
            elif structure_format == "sdf":
                logger.info(f"DEBUG: Reading SDF file: {structure_path}")
                try:
                    mol = Chem.MolFromMolFile(structure_path, removeHs=False)
                except OSError:
                    mol = None
                logger.info(f"DEBUG: MolFromMolFile result: {mol is not None}")
                if mol is None:
                    logger.error(f"Failed to read SDF file: {structure_path}")
                    return False
            elif structure_format == "mol2":
                try:
                    mol = Chem.MolFromMol2File(structure_path, removeHs=False)
                except OSError:
                    mol = None
            else:
                logger.error(f"Unsupported structure format: {structure_format}")
                return False

            if mol is None:
                logger.error(f"Failed to read {structure_format.upper()} file: {structure_path}")
                return False

            # Compute Gasteiger charges (preserves input coordinates)
            AllChem.ComputeGasteigerCharges(mol)
            self._fix_nan_charges(mol)
            logger.info(
                f"Computed Gasteiger charges for {structure_format.upper()} input: {structure_path}"
            )

            # Find ALL halogen atoms
            halogen_indices = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == halogen:
                    halogen_indices.append(atom.GetIdx())

            if not halogen_indices:
                logger.error(f"No {halogen} atom found in {structure_path}")
                return False

            # Create molecule for PDBQT generation
            mol_with_dummy = Chem.RWMol(mol)
            dummy_indices = []

            if add_dummy:
                per_halogen_charge = charge / len(halogen_indices)

                for halogen_idx in halogen_indices:
                    halogen_pos, carbon_pos = self._get_halogen_position(mol, halogen_idx)
                    dummy_pos = self._place_dummy_atom(halogen_pos, carbon_pos, delta_r)

                    # Get the original halogen charge for charge conservation
                    halogen_atom = mol_with_dummy.GetAtomWithIdx(halogen_idx)
                    original_halogen_charge = (
                        halogen_atom.GetDoubleProp("_GasteigerCharge")
                        if halogen_atom.HasProp("_GasteigerCharge")
                        else 0.0
                    )

                    # Add dummy atom (as hydrogen with custom charge)
                    dummy_idx = mol_with_dummy.AddAtom(Chem.Atom(1))  # Hydrogen
                    mol_with_dummy.GetAtomWithIdx(dummy_idx).SetDoubleProp(
                        "dummy_charge", per_halogen_charge
                    )
                    dummy_indices.append(dummy_idx)

                    # Charge conservation: dummy carries +q, so halogen must lose q to keep net charge zero.
                    # q_hal_new = q_hal_original - q_dummy
                    new_halogen_charge = original_halogen_charge - per_halogen_charge
                    halogen_atom.SetDoubleProp("_GasteigerCharge", new_halogen_charge)

                    # Set position of dummy atom
                    conf = mol_with_dummy.GetConformer()
                    conf.SetAtomPosition(dummy_idx, dummy_pos.tolist())

                logger.info(
                    f"Added {len(dummy_indices)} dummy atom(s) for {len(halogen_indices)} {halogen} atom(s)"
                )
            else:
                dummy_indices = None

            # Create PDBQT manually
            self._create_pdbqt_manual(mol_with_dummy, dummy_indices, charge, output_path, add_dummy)

            if add_dummy:
                logger.info(f"Generated PDBQT from {structure_format.upper()}: {output_path}")
            else:
                logger.info(
                    f"Generated control PDBQT from {structure_format.upper()}: {output_path}"
                )
            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error preparing ligand from structure: {e}")
            return False

    def prepare_ligand_from_pdb(
        self,
        pdb_path: str,
        halogen: str,
        charge: float,
        output_path: str,
        delta_r: float = 1.2,
        add_dummy: bool = True,
    ) -> bool:
        """Backward-compatible wrapper around prepare_ligand_from_structure."""
        return self.prepare_ligand_from_structure(
            pdb_path, halogen, charge, output_path, delta_r, add_dummy, structure_format="pdb"
        )

    def batch_generate_ligands(
        self,
        ligands_df: pd.DataFrame,
        output_dir: str,
        smiles_col: str = "smiles",
        halogen_col: str = "halogen",
        charge_col: str = "dummy_charge_e",
        id_col: str = "compound_id",
        add_dummy: bool = True,
    ) -> list[str]:
        """
        Generate PDBQT files for a batch of ligands with optional dummy atom.

        Args:
            ligands_df: DataFrame with ligand information
            output_dir: Directory to save PDBQT files
            smiles_col: Column name for SMILES strings
            halogen_col: Column name for halogen symbols
            charge_col: Column name for dummy atom charges
            id_col: Column name for compound identifiers
            add_dummy: Whether to add dummy atom (True for sigma-hole, False for control)

        Returns:
            List of generated PDBQT file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        for _, row in ligands_df.iterrows():
            try:
                smiles = row[smiles_col]
                halogen = row[halogen_col]
                charge = row[charge_col]
                compound_id = row[id_col]

                # Get halogen-specific delta_r for consistency with charge calculation
                delta_r = self.default_delta_r.get(
                    halogen, 1.2
                )  # Default to 1.2 if halogen not found

                output_path = os.path.join(output_dir, f"{compound_id}_ligand.pdbqt")

                success = self.prepare_ligand_from_smiles(
                    smiles, halogen, charge, output_path, delta_r=delta_r, add_dummy=add_dummy
                )

                if success:
                    generated_files.append(output_path)
                    logger.info(f"Generated ligand for {compound_id}")
                else:
                    logger.error(f"Failed to generate ligand for {compound_id}")

            except Exception as e:
                logger.error(f"Error processing row {row.get(id_col, 'unknown')}: {e}")

        logger.info(f"Generated {len(generated_files)} ligand PDBQT files")
        return generated_files

    def batch_generate_ligands_from_structures(
        self,
        ligands_df: pd.DataFrame,
        structure_dir: str,
        output_dir: str,
        halogen_col: str = "halogen",
        charge_col: str = "dummy_charge_e",
        id_col: str = "compound_id",
        structure_ext: str = ".sdf",
        add_dummy: bool = True,
    ) -> list[str]:
        """
        Generate PDBQT files for a batch of ligands from DFT structure files (PDB/SDF/MOL2).
        Preserves input geometry — no optimization.

        Args:
            ligands_df: DataFrame with ligand information
            structure_dir: Directory containing structure files
            output_dir: Directory to save PDBQT files
            halogen_col: Column name for halogen symbols
            charge_col: Column name for dummy atom charges
            id_col: Column name for compound identifiers
            structure_ext: File extension of structure files (e.g., '.sdf', '.pdb')
            add_dummy: Whether to add dummy atom

        Returns:
            List of generated PDBQT file paths
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        for _, row in ligands_df.iterrows():
            try:
                halogen = row[halogen_col]
                charge = row[charge_col]
                compound_id = row[id_col]

                delta_r = self.default_delta_r.get(halogen, 1.2)

                structure_path = os.path.join(structure_dir, f"{compound_id}{structure_ext}")
                output_path = os.path.join(output_dir, f"{compound_id}_ligand.pdbqt")

                success = self.prepare_ligand_from_structure(
                    structure_path,
                    halogen,
                    charge,
                    output_path,
                    delta_r=delta_r,
                    add_dummy=add_dummy,
                    structure_format="auto",
                )

                if success:
                    generated_files.append(output_path)
                    logger.info(f"Generated ligand for {compound_id}")
                else:
                    logger.error(f"Failed to generate ligand for {compound_id}")

            except Exception as e:
                logger.error(f"Error processing row {row.get(id_col, 'unknown')}: {e}")

        logger.info(f"Generated {len(generated_files)} ligand PDBQT files")
        return generated_files


def example_usage():
    """Example usage of the ligand generator."""
    generator = SigmaHoleLigandGenerator()

    # Example: Iodobenzene
    smiles = "c1ccccc1I"  # Iodobenzene
    halogen = "I"
    charge = 0.0611  # Example charge from Vmax=26.0 kcal/mol

    # Generate ligand WITH dummy atom (for sigma-hole modeling)
    output_file_sigma = "iodobenzene_sigma.pdbqt"
    print(f"Generating ligand PDBQT FOR SIGMA-HOLE: {smiles} with {halogen} dummy charge {charge}")
    success_sigma = generator.prepare_ligand_from_smiles(
        smiles, halogen, charge, output_file_sigma, add_dummy=True
    )

    # Generate ligand WITHOUT dummy atom (for control experiments)
    output_file_control = "iodobenzene_control.pdbqt"
    print(f"Generating ligand PDBQT FOR CONTROL: {smiles} (NO DUMMY ATOM)")
    success_control = generator.prepare_ligand_from_smiles(
        smiles, halogen, charge, output_file_control, add_dummy=False
    )

    if success_sigma:
        print(f"Successfully generated sigma-hole ligand: {output_file_sigma}")
        # Show first few lines of the generated file
        try:
            with open(output_file_sigma, "r") as f:
                lines = f.readlines()
            print("First 10 lines of sigma-hole PDBQT:")
            for line in lines[:10]:
                print(line.rstrip())
        except Exception as e:
            print(f"Could not read generated file: {e}")
    else:
        print("Failed to generate sigma-hole ligand PDBQT")

    if success_control:
        print(f"Successfully generated control ligand: {output_file_control}")
        # Show first few lines of the generated file
        try:
            with open(output_file_control, "r") as f:
                lines = f.readlines()
            print("First 10 lines of control PDBQT:")
            for line in lines[:10]:
                print(line.rstrip())
        except Exception as e:
            print(f"Could not read generated file: {e}")
    else:
        print("Failed to generate control ligand PDBQT")

    return success_sigma and success_control


if __name__ == "__main__":
    example_usage()
