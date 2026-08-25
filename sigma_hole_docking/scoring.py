"""
Sigma Hole Docking Scoring Module

Contains physics-based scoring functionality for sigma-hole interactions.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _calculate_pairwise_energy(ligand_atoms: List[Dict], receptor_atoms: List[Dict]) -> float:
    """
    Calculate pairwise energy between ligand and receptor atoms (helper for optimization).
    Uses the same physics as calculate_physics_score but without alignment/separation.
    Includes directional Coulomb corrections for sigma-hole interactions.
    """
    total_energy = 0.0
    pairs_count = 0

    # Calculate interactions between all ligand and receptor atoms
    for lig_atom in ligand_atoms:
        for rec_atom in receptor_atoms:
            # Calculate distance
            dx = lig_atom['x'] - rec_atom['x']
            dy = lig_atom['y'] - rec_atom['y']
            dz = lig_atom['z'] - rec_atom['z']
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Skip if too far apart (to avoid negligible interactions)
            if distance > 6.0:
                continue

            # Determine charge scale factor for directional corrections (default: no correction)
            charge_factor = 1.0
            _halogen = None
            _bonded_c = None

            # Check if lig_atom is a halogen
            if lig_atom['element'] in ['F', 'Cl', 'Br', 'I', 'At']:
                _halogen = lig_atom
                # Find bonded carbon to this halogen
                if ligand_atoms:
                    min_dist = float("inf")
                    for carbon in ligand_atoms:
                        if carbon['element'] == "C":
                            dist = np.sqrt(
                                (lig_atom["x"] - carbon["x"])**2 +
                                (lig_atom["y"] - carbon["y"])**2 +
                                (lig_atom["z"] - carbon["z"])**2
                            )
                            if dist < min_dist:
                                min_dist = dist
                                _bonded_c = carbon
            # Check if lig_atom is carbon bonded to a halogen
            elif lig_atom['element'] == "C":
                # Find bonded halogen to this carbon
                min_dist = float("inf")
                for hydrogen in ligand_atoms:  # Actually looking for halogen
                    if hydrogen['element'] in ['F', 'Cl', 'Br', 'I', 'At']:
                        dist = np.sqrt(
                            (lig_atom["x"] - hydrogen["x"])**2 +
                            (lig_atom["y"] - hydrogen["y"])**2 +
                            (lig_atom["z"] - hydrogen["z"])**2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            _halogen = hydrogen
                            _bonded_c = lig_atom

            # Initialize energy variables to avoid UnboundLocalError
            lj_energy = 0.0
            coulomb_energy = 0.0

            # Lennard-Jones (Van der Waals)
            is_lig_dummy = lig_atom.get('is_dummy', False)
            is_rec_dummy = rec_atom.get('is_dummy', False)

            if is_lig_dummy or is_rec_dummy:
                epsilon, sigma = 0.02, 1.2
            else:
                epsilon, sigma = _get_lj_parameters(
                    lig_atom['element'], rec_atom['element']
                )

            # Prevent division by zero or excessively small distances
            min_dist_clamp = max(0.6 * sigma, 0.5)
            if distance < min_dist_clamp:
                distance = min_dist_clamp

            if distance > 0:
                lj_ratio = sigma / distance
                lj_term = lj_ratio ** 6
                lj_energy = 4.0 * epsilon * (lj_term * lj_term - lj_term)
                if lj_energy > 10.0:
                    lj_energy = 10.0
            else:
                lj_energy = 0.0

            # Coulomb (Electrostatics) with directional corrections
            if distance > 0:
                # Note: dielectric_coeff would need to be passed or accessed from instance
                # For now, using default of 1.0 (gas phase) - this should be adjusted
                # when integrating with the main class
                epsilon_r = 1.0  # Default to gas phase

                # Determine charge scale factor for this pair
                if lig_atom is _halogen:
                    # Halogen-acceptor: suppress in sigma-hole direction
                    angle = _compute_cx_acceptor_angle(_halogen, rec_atom, ligand_atoms)
                    charge_factor = _halogen_acceptor_charge_scale(angle)
                elif _bonded_c is not None and lig_atom is _bonded_c:
                    # Bonded carbon-acceptor: suppress in sigma-hole direction
                    angle = _compute_cx_acceptor_angle(_halogen, rec_atom, ligand_atoms)
                    charge_factor = _bonded_carbon_charge_scale(angle)
                elif is_lig_dummy:
                    # Dummy atom: only interact with electronegative acceptors
                    charge_factor = _dummy_acceptor_charge_scale(rec_atom['element'])

                # Note: k_coulomb would need to be passed or accessed from instance
                # For now, using default value - this should be adjusted when integrating
                k_coulomb = 332.06  # Default value
                coulomb_energy = (k_coulomb *
                    lig_atom['charge'] *
                    rec_atom['charge'] /
                    (epsilon_r * distance)) * charge_factor
            else:
                coulomb_energy = 0.0

            # Total pairwise energy
            pair_energy = lj_energy + coulomb_energy
            total_energy += pair_energy
            pairs_count += 1

    return total_energy


def _get_lj_parameters(atom1: str, atom2: str) -> Tuple[float, float]:
    """
    Get Lennard-Jones parameters for an atom pair.
    Uses Lorentz-Berthelot mixing rules if specific params not available.
    Dummy atoms (hydrogens with positive charge) have zero LJ parameters
    as they represent virtual charge sites, not physical atoms.
    """
    # Check if either atom is a dummy hydrogen (H with positive charge)
    # Note: This check assumes the calling code will identify dummy atoms
    # by their charge > 0.01. For now, we'll implement a simple version
    # that treats all hydrogens as having normal LJ parameters, and
    # the caller should handle dummy atom special case.
    #
    # TODO: To properly identify dummy atoms, we need to pass charge information
    # to this method or store it in the atom dict. For now, we rely on the
    # fact that dummy atoms should have very small LJ interactions anyway.

    # Try direct lookup (sorted to handle atom1-atom2 vs atom2-atom1)
    key1 = (atom1, atom2)
    key2 = (atom2, atom1)

    # These would need to be accessed from the instance or passed as parameters
    # For now, using default parameters
    defaults = {
        'H': (0.05, 1.5),
        'C': (0.10, 2.0),
        'N': (0.10, 1.8),
        'O': (0.15, 1.8),
        'S': (0.20, 2.0),
        'F': (0.15, 1.7),
        'Cl': (0.20, 2.0),
        'Br': (0.22, 2.1),
        'I': (0.25, 2.2),
    }

    eps1, sig1 = defaults.get(atom1, (0.10, 2.0))
    eps2, sig2 = defaults.get(atom2, (0.10, 2.0))

    epsilon = math.sqrt(eps1 * eps2)
    sigma = (sig1 + sig2) / 2

    return (epsilon, sigma)


def _compute_cx_acceptor_angle(halogen_atom, acceptor_atom, ligand_atoms):
    """Compute C-X...Acceptor angle at the halogen vertex.
    Returns angle in degrees, or None if bonded carbon not found."""
    hal_pos = np.array([halogen_atom['x'], halogen_atom['y'], halogen_atom['z']])
    hal_elem = halogen_atom['element']
    cutoff = {'F': 1.8, 'Cl': 2.2, 'Br': 2.4, 'I': 2.6, 'At': 2.7}.get(hal_elem, 2.3) + 0.3
    min_c_dist = float('inf')
    carbon_atom = None
    for atom in ligand_atoms:
        if atom['element'] == 'C':
            c_pos = np.array([atom['x'], atom['y'], atom['z']])
            d = np.linalg.norm(hal_pos - c_pos)
            if d < min_c_dist and d < cutoff:
                min_c_dist = d
                carbon_atom = atom
    if carbon_atom is None:
        return None
    vec_xc = np.array([carbon_atom['x'] - halogen_atom['x'],
                       carbon_atom['y'] - halogen_atom['y'],
                       carbon_atom['z'] - halogen_atom['z']])
    vec_xa = np.array([acceptor_atom['x'] - halogen_atom['x'],
                       acceptor_atom['y'] - halogen_atom['y'],
                       acceptor_atom['z'] - halogen_atom['z']])
    norm_xc = np.linalg.norm(vec_xc)
    norm_xa = np.linalg.norm(vec_xa)
    if norm_xc < 1e-8 or norm_xa < 1e-8:
        return None
    cos_angle = np.dot(vec_xc, vec_xa) / (norm_xc * norm_xa)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _halogen_acceptor_charge_scale(angle_deg):
    """Directional Coulomb correction for halogen-acceptor.
    In the sigma-hole direction (angle > 140), the halogen's negative
    charge should be suppressed - the dummy atom handles this direction.
    angle > 140: 0.0, angle < 90: 1.0, linear between."""
    if angle_deg is None:
        return 1.0
    if angle_deg >= 140.0:
        return 0.0
    elif angle_deg <= 90.0:
        return 1.0
    else:
        return 1.0 - (angle_deg - 90.0) / (140.0 - 90.0)


def _bonded_carbon_charge_scale(angle_deg):
    """Directional Coulomb correction for bonded carbon.
    The C's positive charge and the sigma-hole are the same polarization.
    In sigma-hole direction this is double-counting; suppress it.
    angle > 140: 0.0, angle < 90: 1.0, linear between."""
    if angle_deg is None:
        return 1.0
    if angle_deg >= 140.0:
        return 0.0
    elif angle_deg <= 90.0:
        return 1.0
    else:
        return 1.0 - (angle_deg - 90.0) / (140.0 - 90.0)


def _dummy_acceptor_charge_scale(rec_element):
    """Scale dummy atom Coulomb by receptor atom type.
    The sigma-hole only attracts electronegative acceptors (O, N, S, F).
    Other receptor atoms should not interact with the dummy."""
    if rec_element in ['O', 'N', 'S', 'F']:
        return 1.0
    else:
        return 0.0


def _find_bonded_carbon(halogen_atom, ligand_atoms):
    """Find the carbon atom bonded to the halogen."""
    hal_pos = np.array([halogen_atom['x'], halogen_atom['y'], halogen_atom['z']])
    hal_elem = halogen_atom['element']
    cutoff = {'F': 1.8, 'Cl': 2.2, 'Br': 2.4, 'I': 2.6, 'At': 2.7}.get(hal_elem, 2.3) + 0.3
    min_c_dist = float('inf')
    carbon_atom = None
    for atom in ligand_atoms:
        if atom['element'] == 'C':
            c_pos = np.array([atom['x'], atom['y'], atom['z']])
            d = np.linalg.norm(hal_pos - c_pos)
            if d < min_c_dist and d < cutoff:
                min_c_dist = d
                carbon_atom = atom
    return carbon_atom