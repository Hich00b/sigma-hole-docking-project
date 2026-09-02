"""
Geometry Validator

Validates molecular geometry against DFT reference values.
Checks C-X bond lengths, C-X-C bond angles, and halogen atom counts.
Used for quality control of DFT-optimized structure input.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

# Reference C-X bond lengths (B3LYP/6-311G(d,p)) in Angstroms
REFERENCE_CX_LENGTHS = {
    "F": 1.35,
    "Cl": 1.78,
    "Br": 1.94,
    "I": 2.14,
    "At": 2.20,
}

# Expected C-X-C bond angles based on carbon hybridization
REFERENCE_CXC_ANGLES = {
    "sp3": 109.5,
    "sp2": 120.0,
    "sp": 180.0,
}

# Tolerances for validation
BOND_LENGTH_TOLERANCE = 0.10  # Angstroms
BOND_ANGLE_TOLERANCE = 15.0  # degrees

# Supported formats
SUPPORTED_FORMATS = [".pdb", ".sdf", ".mol", ".mol2"]


class GeometryValidator:
    """
    Validates molecular geometry against DFT reference values.

    Ensures structure files used in the pipeline match the geometry
    on which Vmax calculations were performed.
    """

    def __init__(
        self,
        bond_tolerance: float = BOND_LENGTH_TOLERANCE,
        angle_tolerance: float = BOND_ANGLE_TOLERANCE,
    ):
        """
        Initialize the validator.

        Args:
            bond_tolerance: Allowed deviation from reference C-X bond length (A)
            angle_tolerance: Allowed deviation from expected C-X-C angle (degrees)
        """
        self.bond_tolerance = bond_tolerance
        self.angle_tolerance = angle_tolerance

    def _read_structure(self, structure_path: str, structure_format: str = "auto"):
        """Read a structure file and return an RDKit molecule."""
        if structure_format == "auto":
            ext = os.path.splitext(structure_path)[1].lower()
            format_map = {".pdb": "pdb", ".sdf": "sdf", ".mol": "sdf", ".mol2": "mol2"}
            structure_format = format_map.get(ext, "")

        if structure_format == "pdb":
            try:
                mol = Chem.MolFromPDBFile(structure_path, removeHs=False)
            except OSError:  # Catch-all for PDB file reading errors
                mol = None
        elif structure_format in ("sdf", "mol"):
            try:
                mol = Chem.MolFromMolFile(structure_path, removeHs=False)
            except OSError:  # Catch-all for SDF/MOL file reading errors
                mol = None
            if mol is None:
                logger.error(f"Failed to read SDF/MOL file: {structure_path}")
        elif structure_format == "mol2":
            try:
                mol = Chem.MolFromMol2File(structure_path, removeHs=False)
            except OSError:  # Catch-all for MOL2 file reading errors
                mol = None
        else:
            return None

        return mol

    def _detect_carbon_hybridization(self, carbon_idx: int, mol: Chem.Mol) -> str:
        carbon = mol.GetAtomWithIdx(carbon_idx)
        hyb = carbon.GetHybridization()
        if hyb == Chem.HybridizationType.SP3:
            return "sp3"
        elif hyb in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP2D):
            return "sp2"
        elif hyb == Chem.HybridizationType.SP:
            return "sp"
        else:
            return "sp3"  # safe default

    def _calculate_bond_length(self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> float:
        """Calculate distance between two atoms."""
        conf = mol.GetConformer()
        p1 = np.array(conf.GetAtomPosition(atom1_idx))
        p2 = np.array(conf.GetAtomPosition(atom2_idx))
        return float(np.linalg.norm(p1 - p2))

    def _calculate_angle(
        self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int, atom3_idx: int
    ) -> float:
        """Calculate C-X-C bond angle in degrees."""
        conf = mol.GetConformer()
        v1 = np.array(conf.GetAtomPosition(atom1_idx))
        v2 = np.array(conf.GetAtomPosition(atom2_idx))
        v3 = np.array(conf.GetAtomPosition(atom3_idx))

        vec1 = v1 - v2
        vec2 = v3 - v2

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def validate_molecule_geometry(self, mol: Chem.Mol, halogen: str) -> dict:
        """
        Validate geometry of a single molecule.

        Args:
            mol: RDKit molecule object (with 3D coordinates)
            halogen: Halogen symbol to validate

        Returns:
            Dictionary with validation results
        """
        result = {
            "halogen": halogen,
            "halogen_count": 0,
            "bonds_valid": True,
            "angles_valid": True,
            "bond_details": [],
            "angle_details": [],
            "overall_valid": True,
            "messages": [],
            "error": None,
        }

        # Find all halogen atoms
        halogen_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == halogen]
        result["halogen_count"] = len(halogen_indices)

        if not halogen_indices:
            result["error"] = f"No {halogen} atoms found"
            result["overall_valid"] = False
            return result

        ref_length = REFERENCE_CX_LENGTHS.get(halogen, 2.14)

        for halogen_idx in halogen_indices:
            halogen_atom = mol.GetAtomWithIdx(halogen_idx)

            # Find bonded carbon
            carbon_neighbors = [
                nbr for nbr in halogen_atom.GetNeighbors() if nbr.GetAtomicNum() == 6
            ]

            if not carbon_neighbors:
                result["messages"].append(
                    f"No carbon neighbor for {halogen} at index {halogen_idx}"
                )
                continue

            carbon = carbon_neighbors[0]
            carbon_idx = carbon.GetIdx()

            # Check C-X bond length
            bond_length = self._calculate_bond_length(mol, carbon_idx, halogen_idx)
            bond_error = abs(bond_length - ref_length)
            bond_ok = bond_error <= self.bond_tolerance

            result["bond_details"].append(
                {
                    "halogen_idx": halogen_idx,
                    "carbon_idx": carbon_idx,
                    "measured": round(bond_length, 3),
                    "reference": ref_length,
                    "error": round(bond_error, 3),
                    "passed": bond_ok,
                }
            )

            if not bond_ok:
                result["bonds_valid"] = False
                result["messages"].append(
                    f"C-{halogen} bond length {bond_length:.3f} A deviates "
                    f"by {bond_error:.3f} A from reference {ref_length:.2f} A"
                )

            # Check C-X-C angle (find second carbon in the chain)
            carbon_neighbors_2 = [
                nbr
                for nbr in carbon.GetNeighbors()
                if nbr.GetAtomicNum() == 6 and nbr.GetIdx() != halogen_idx
            ]

            if carbon_neighbors_2:
                carbon2 = carbon_neighbors_2[0]
                carbon2_idx = carbon2.GetIdx()

                angle = self._calculate_angle(mol, halogen_idx, carbon_idx, carbon2_idx)
                hybridization = self._detect_carbon_hybridization(carbon_idx, mol)
                ref_ang = REFERENCE_CXC_ANGLES.get(hybridization, 109.5)
                angle_error = abs(angle - ref_ang)
                angle_ok = angle_error <= self.angle_tolerance

                result["angle_details"].append(
                    {
                        "measured": round(angle, 1),
                        "expected": ref_ang,
                        "hybridization": hybridization,
                        "error": round(angle_error, 1),
                        "passed": angle_ok,
                    }
                )

                if not angle_ok:
                    result["angles_valid"] = False
                    result["messages"].append(
                        f"C-{halogen}-C angle {angle:.1f} deg deviates by {angle_error:.1f} deg "
                        f"from expected {ref_ang:.1f} deg ({hybridization})"
                    )

        result["overall_valid"] = result["bonds_valid"] and result["angles_valid"]
        return result

    def validate_structure_file(
        self, structure_path: str, halogen: str, structure_format: str = "auto"
    ) -> dict:
        """
        Validate a single structure file.

        Args:
            structure_path: Path to structure file
            halogen: Halogen symbol to look for
            structure_format: 'auto', 'pdb', 'sdf', 'mol2'

        Returns:
            Validation results dictionary
        """
        result = {
            "file": structure_path,
            "halogen": halogen,
            "valid": False,
            "error": None,
            "details": None,
        }

        mol = self._read_structure(structure_path, structure_format)
        if mol is None:
            result["error"] = f"Failed to read {structure_path}"
            return result

        try:
            mol_3d = mol.GetConformer()
        except (OSError, ValueError, RuntimeError) as e:  # Catch-all for conformer retrieval errors
            result["error"] = f"Failed to get conformer: {e}"
            return result

        if mol_3d.Is3D() or mol_3d.GetNumAtoms() > 0:
            try:
                details = self.validate_molecule_geometry(mol, halogen)
                details["file"] = structure_path
                result["details"] = details
                result["valid"] = details["overall_valid"]
            except (OSError, ValueError, RuntimeError) as e:  # Catch-all for errors during geometry validation
                result["error"] = f"Error during geometry validation: {e}"
                return result
        else:
            result["error"] = "No 3D coordinates found in molecule"

        return result

    def batch_validate(
        self,
        structure_dir: str,
        input_csv: str,
        halogen_col: str = "halogen",
        id_col: str = "compound_id",
        structure_ext: str = ".sdf",
    ) -> pd.DataFrame:
        """
        Validate a batch of structure files from a directory.

        Args:
            structure_dir: Directory containing structure files
            input_csv: Pipeline input CSV with compound information
            halogen_col: Column for halogen symbols
            id_col: Column for compound identifiers
            structure_ext: File extension (e.g., '.sdf', '.pdb')

        Returns:
            DataFrame with validation results for each compound
        """
        df_input = pd.read_csv(input_csv)
        results = []

        for _, row in df_input.iterrows():
            compound_id = row[id_col]
            halogen = row[halogen_col]
            structure_path = os.path.join(structure_dir, f"{compound_id}{structure_ext}")

            val_result = self.validate_structure_file(structure_path, halogen)
            val_result["compound_id"] = compound_id
            results.append(val_result)

        return pd.DataFrame(results)

    def generate_validation_report(self, validation_df: pd.DataFrame, output_path: str) -> str:
        """
        Generate a summary validation report.

        Args:
            validation_df: DataFrame with validation results
            output_path: Path to save the report

        Returns:
            Path to the generated report file
        """
        total = len(validation_df)
        valid = validation_df["valid"].sum() if "valid" in validation_df.columns else 0
        invalid = total - valid

        report_lines = [
            "=" * 60,
            "Geometry Validation Report",
            "=" * 60,
            f"Total files checked: {total}",
            f"Valid: {valid} ({100 * valid / total:.1f}%)",
            f"Invalid: {invalid} ({100 * invalid / total:.1f}%)",
            "=" * 60,
            "",
        ]

        # Bond length stats
        if "details" in validation_df.columns:
            all_bonds = []
            all_angles = []
            for _, row in validation_df.iterrows():
                if row.get("details"):
                    all_bonds.extend(row["details"].get("bond_details", []))
                    all_angles.extend(row["details"].get("angle_details", []))

            if all_bonds:
                errors = [b["error"] for b in all_bonds]
                report_lines.append("C-X Bond Length Statistics:")
                report_lines.append(f"  Mean error: {np.mean(errors):.3f} A")
                report_lines.append(f"  Max error:  {np.max(errors):.3f} A")
                report_lines.append(
                    f"  Passed: {sum(1 for e in errors if e <= self.bond_tolerance)}/{len(errors)}"
                )
                report_lines.append("")

            if all_angles:
                errors = [a["error"] for a in all_angles]
                report_lines.append("C-X-C Angle Statistics:")
                report_lines.append(f"  Mean error: {np.mean(errors):.1f} deg")
                report_lines.append(f"  Max error:  {np.max(errors):.1f} deg")
                report_lines.append(
                    f"  Passed: {sum(1 for e in errors if e <= self.angle_tolerance)}/{len(errors)}"
                )
                report_lines.append("")

        # Failed files
        failed = (
            validation_df[~validation_df["valid"]]
            if "valid" in validation_df.columns
            else pd.DataFrame()
        )
        if not failed.empty:
            report_lines.append("Failed Validations:")
            for _, row in failed.iterrows():
                compound_id = row.get("compound_id", "unknown")
                error = row.get("error") or row.get("details", {}).get("messages", ["Unknown"])
                if isinstance(error, list):
                    error = "; ".join(error)
                report_lines.append(f"  {compound_id}: {error}")
            report_lines.append("")

        report_lines.append("=" * 60)

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Validation report saved to {output_path}")
        return output_path


def example_usage():
    """Example usage of the geometry validator."""
    validator = GeometryValidator()

    # Validate a single file
    # result = validator.validate_structure_file("path/to/molecule.sdf", "I")
    # print(f"Valid: {result['valid']}")

    # Batch validate
    # df_results = validator.batch_validate("structures/", "input.csv", halogen_col="halogen")
    # print(df_results)

    print("GeometryValidator example — configure paths to use")
    return validator


if __name__ == "__main__":
    example_usage()
