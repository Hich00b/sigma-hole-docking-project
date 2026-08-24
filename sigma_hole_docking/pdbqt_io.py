"""
Sigma Hole PDBQT I/O Module

Consolidated PDBQT parsing and writing functionality for the sigma-hole docking pipeline.
Provides common functions for reading and writing PDBQT files used across ligand generation,
receptor processing, docking, and results analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parse_pdbqt(pdbqt_path: str) -> List[Dict]:
    """
    Parse PDBQT file to extract atom information.
    Handles both ATOM and HETATM records (for OpenBabel compatibility).

    Returns:
        List of dictionaries with keys: 'element', 'x', 'y', 'z', 'charge', 'is_dummy'
    """
    atoms = []
    parsing_errors = 0

    try:
        with open(pdbqt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # PDBQT format:
                    # ATOM      1  I   LIG B   1       0.000   0.000   0.000  0.00  0.00    -0.100 I
                    parts = line.split()
                    if len(parts) >= 12:
                        try:
                            atom_index = int(parts[1])
                            # In PDBQT format:
                            # parts[2] = atom name (often the element symbol, e.g., "C", "O", "I")
                            # parts[12] = atom type (can be different, e.g., "I3" for sigma-hole iodine)
                            element = parts[2]  # Element symbol is at index 2 (3rd field)
                            atom_type = parts[12] if len(parts) > 12 else element  # Atom type is at index 12 (13th field) if present

                            # Normalize halogen element names (handle cases like "CL" -> "Cl")
                            if element.upper() == 'CL':
                                element = 'Cl'
                            elif element.upper() == 'BR':
                                element = 'Br'
                            elif element.upper() == 'I':
                                element = 'I'  # Already correct, but explicit for clarity
                            # Note: Hydrogen "H" doesn't need normalization

                            x = float(parts[6])
                            y = float(parts[7])
                            z = float(parts[8])
                            # occupancy = float(parts[9])
                            # temperature_factor = float(parts[10])  # Usually 0.00 in PDBQT
                            charge = float(parts[11])  # Charge is at index 11 in PDBQT format
                            # Determine if this is a dummy atom (virtual charge site)
                            # Dummy atoms often have atom_type starting with 'EP' or have element H with positive charge
                            is_dummy = (atom_type == "EP")

                            atoms.append({
                                'index': atom_index,
                                'element': element,
                                'x': x, 'y': y, 'z': z,
                                'charge': charge,
                                'atom_type': atom_type,
                                'is_dummy': is_dummy
                            })
                        except (ValueError, IndexError) as e:
                            parsing_errors += 1
                            if parsing_errors <= 5:  # Limit error messages to avoid spam
                                logger.debug(f"Could not parse ATOM/HETATM line {line_num}: {line}")
                            continue
                    else:
                        parsing_errors += 1
                        if parsing_errors <= 5:
                            logger.debug(f"Malformed ATOM/HETATM line {line_num} (too few fields): {line}")

        if parsing_errors > 5:
            logger.debug(f"... and {parsing_errors - 5} more parsing errors")

    except FileNotFoundError:
        logger.error(f"PDBQT file not found: {pdbqt_path}")
        return []  # Return empty list to signal failure
    except Exception as e:
        logger.error(f"Error reading PDBQT file {pdbqt_path}: {e}")
        return []  # Return empty list to signal failure

    # Validate that we parsed some atoms
    if not atoms:
        logger.warning(f"No ATOM or HETATM records found in PDBQT file: {pdbqt_path}")
        return atoms  # Return empty list - caller should handle this

    logger.debug(f"Parsed {len(atoms)} atoms from {pdbqt_path}")
    return atoms


def parse_pdbqt_detailed(pdbqt_path: str) -> List[Dict]:
    """
    Parse PDBQT file to extract detailed atom information for validation purposes.

    Returns:
        List of dictionaries with atomic details: 'index', 'element', 'x', 'y', 'z', 'charge'
    """
    atoms = []

    try:
        with open(pdbqt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            atom_data = {
                                'index': int(parts[1]),
                                'element': parts[2],
                                'x': float(parts[6]),
                                'y': float(parts[7]),
                                'z': float(parts[8]),
                                'charge': float(parts[10]) if len(parts) > 10 else 0.0
                            }
                            atoms.append(atom_data)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logger.error(f"Error parsing PDBQT {pdbqt_path}: {e}")

    return atoms


def write_pdbqt_atoms(atoms: List[Dict], output_path: str,
                     title: str = "Generated by Sigma Hole PDBQT I/O",
                     is_docking: bool = False) -> bool:
    """
    Write atoms to PDBQT file format.

    Args:
        atoms: List of atom dictionaries with keys: 'element', 'x', 'y', 'z', 'charge', optionally 'is_dummy'
        output_path: Path to save PDBQT file
        title: Title/remark for the PDBQT file
        is_docking: Whether this is for docking (adds ROOT/ENDROOT/TORSDOF sections)

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w') as f:
            f.write(f"REMARK  {title}\n")

            if is_docking:
                f.write("ROOT\n")

            for i, atom in enumerate(atoms):
                element = atom.get('element', '')
                x = atom.get('x', 0.0)
                y = atom.get('y', 0.0)
                z = atom.get('z', 0.0)
                charge = atom.get('charge', 0.0)

                # Determine atom type - use element for most atoms, EP for dummy atoms
                is_dummy = atom.get('is_dummy', False)
                if is_dummy:
                    atom_type = 'EP'  # Extra point for dummy atoms
                else:
                    atom_type = element  # Use element as atom type

                if is_docking:
                    f.write(f"ATOM {i+1:4d} {element:<2s} LIG B 1 "
                            f"{x:8.3f}{y:8.3f}{z:8.3f} "
                            f"0.00 0.00 {charge:7.4f} {atom_type:2s}\n")
                else:
                    # Simple format for receptor/ligand files without docking sections
                    f.write(f"ATOM  {i+1:4d} {element:<2s}      {x:8.3f}{y:8.3f}{z:8.3f} {charge:7.4f}\n")

            if is_docking:
                f.write("ENDROOT\n")
                f.write("TORSDOF\n")
                # Count rotatable bonds (simplified - would need molecular topology for accurate count)
                f.write("0\n")

        logger.info(f"Wrote {len(atoms)} atoms to PDBQT file: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing PDBQT file {output_path}: {e}")
        return False


def write_pdbqt_from_mol(mol, output_path: str,
                        title: str = "Generated by Sigma Hole PDBQT I/O",
                        is_docking: bool = False,
                        dummy_indices: Optional[List[int]] = None,
                        total_dummy_charge: float = 0.0) -> bool:
    """
    Write PDBQT file from RDKit molecule object.

    Args:
        mol: RDKit molecule object (can be ROMol or RWMol)
        output_path: Path to save PDBQT file
        title: Title/remark for the PDBQT file
        is_docking: Whether to include docking sections (ROOT/ENDROOT/TORSDOF)
        dummy_indices: List of dummy atom indices (if any)
        total_dummy_charge: Total charge for all dummy atoms combined

    Returns:
        True if successful, False otherwise
    """
    try:
        from rdkit import Chem

        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        # Normalize dummy_indices to a set for fast lookup
        dummy_set = set(dummy_indices) if dummy_indices is not None else set()

        with open(output_path, 'w') as f:
            f.write(f"REMARK  {title}\n")

            if is_docking:
                f.write("ROOT\n")

            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)

                # Determine atom type and charge
                element = atom.GetSymbol()
                if i in dummy_set:
                    # Dummy atom: use EP type
                    atom_type = 'EP'
                    # Distribute total dummy charge equally among all dummy atoms
                    atom_charge = total_dummy_charge / len(dummy_set) if len(dummy_set) > 0 else 0.0
                else:
                    # Use standard atom types
                    atom_type = element
                    # Get Gasteiger partial charge if available
                    atom_charge = (
                        atom.GetDoubleProp('_GasteigerCharge')
                        if atom.HasProp('_GasteigerCharge')
                        else 0.0
                    )

                if is_docking:
                    f.write(f"ATOM {i+1:4d} {element:<2s} LIG B 1 "
                            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f} "
                            f"0.00 0.00 {atom_charge:7.4f} {atom_type:2s}\n")
                else:
                    f.write(f"ATOM  {i+1:4d} {element:<2s}      {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f} {atom_charge:7.4f}\n")

            if is_docking:
                f.write("ENDROOT\n")
                f.write("TORSDOF\n")
                # Count rotatable bonds (would need to compute from romol)
                try:
                    from rdkit.Chem import AllChem
                    rotatable_bonds = AllChem.CalcNumRotatableBonds(mol)
                    f.write(f"{rotatable_bonds}\n")
                except:
                    f.write("0\n")

        if is_docking:
            logger.info(f"Wrote PDBQT with {len(dummy_set)} dummy atom(s): {output_path}")
        else:
            logger.info(f"Wrote PDBQT: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing PDBQT from molecule {output_path}: {e}")
        return False


def compute_geometric_center(atoms: List[Dict]) -> Tuple[float, float, float]:
    """
    Compute the geometric center of a set of atoms.

    Args:
        atoms: List of atom dictionaries with 'x', 'y', 'z' keys

    Returns:
        Tuple of (x, y, z) coordinates of the geometric center
    """
    if not atoms:
        return 0.0, 0.0, 0.0

    sum_x = sum(atom['x'] for atom in atoms)
    sum_y = sum(atom['y'] for atom in atoms)
    sum_z = sum(atom['z'] for atom in atoms)
    count = len(atoms)

    return sum_x / count, sum_y / count, sum_z / count


def compute_distance(atom1: Dict, atom2: Dict) -> float:
    """
    Compute Euclidean distance between two atoms.

    Args:
        atom1: First atom dictionary with 'x', 'y', 'z' keys
        atom2: Second atom dictionary with 'x', 'y', 'z' keys

    Returns:
        Distance in Angstroms
    """
    dx = atom1['x'] - atom2['x']
    dy = atom1['y'] - atom2['y']
    dz = atom1['z'] - atom2['z']
    return np.sqrt(dx*dx + dy*dy + dz*dz)