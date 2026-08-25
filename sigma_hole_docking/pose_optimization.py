"""
Sigma Hole Docking Pose Optimization Module

Contains pose optimization functionality for sigma-hole interactions.
"""

import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _validate_coordinates(atoms: List[Dict], context: str = "") -> bool:
    """
    Validate that all coordinates are finite numbers (not NaN or Inf).

    Args:
        atoms: List of atom dictionaries with x, y, z coordinates
        context: Context string for logging (e.g., "after alignment", "after optimization")

    Returns:
        True if all coordinates are valid, False otherwise
    """
    if not atoms:
        if context:
            logger.warning(f"VALIDATION: No atoms to validate {context}")
        return True  # Empty list is technically valid

    for i, atom in enumerate(atoms):
        for coord_name in ['x', 'y', 'z']:
            coord_val = atom[coord_name]
            if not isinstance(coord_val, (int, float)) or not np.isfinite(coord_val):
                logger.error(f"INVALID COORDINATES {context}: Atom {i} ({atom.get('element', '?')}) has {coord_name}={coord_val}")
                return False
    return True


def _local_optimize_pose(ligand_atoms: List[Dict], receptor_atoms: List[Dict]) -> List[Dict]:
    """
    Perform local optimization to refine the sigma-hole pose and find the energy minimum.

    Optimizes over:
    - Translation along the halogen-oxygen (X···O) axis
    - Rotation around an axis perpendicular to both C-X and X···O
    - Translation in two directions perpendicular to the X···O axis
    - Rotation around the X···O axis

    This expanded search allows the ligand to avoid repulsive clashes while
    maintaining approximately correct sigma-hole geometry.

    Returns the pose with the lowest (most negative) energy.
    """
    # Make a copy to work with
    best_atoms = copy.deepcopy(ligand_atoms)
    # Note: _calculate_pairwise_energy would need to be imported from scoring module
    # For now, we'll assume it's available or pass it as a parameter
    # best_energy = _calculate_pairwise_energy(best_atoms, receptor_atoms)
    best_energy = float('inf')  # Placeholder - will be updated when imported

    # Find key atoms for defining the optimization axes - handle multiple halogens
    # Note: _find_halogen_and_carbon would need to be imported from alignment module
    # Note: _find_acceptor_atoms would need to be imported from alignment module
    halogen_carbon_pairs = []  # Placeholder
    acceptor_oxygens = []  # Placeholder

    if not acceptor_oxygens:
        # Can't optimize without acceptor atoms
        return best_atoms

    # Filter out pairs where halogen is None (shouldn't happen based on implementation, but safe)
    valid_pairs = [(h, c) for h, c in halogen_carbon_pairs if h is not None]
    if not valid_pairs:
        # Can't optimize without halogen atoms
        return best_atoms

    # Separate pairs with and without carbon
    pairs_with_carbon = [(h, c) for h, c in valid_pairs if c is not None]
    pairs_without_carbon = [(h, c) for h, c in valid_pairs if c is None]

    # Select best halogen-carbon pair based on distance to oxygen
    if pairs_with_carbon:
        # Find the halogen-carbon pair where halogen is closest to any oxygen
        best_pair = None
        min_halogen_to_oxygen_dist = float('inf')

        for halogen_atom, carbon_atom in pairs_with_carbon:
            # Find distance from this halogen to closest oxygen
            min_dist = float('inf')
            for oxygen in acceptor_oxygens:
                dist = np.sqrt(
                    (halogen_atom['x'] - oxygen['x'])**2 +
                    (halogen_atom['y'] - oxygen['y'])**2 +
                    (halogen_atom['z'] - oxygen['z'])**2
                )
                if dist < min_dist:
                    min_dist = dist

            if min_dist < min_halogen_to_oxygen_dist:
                min_halogen_to_oxygen_dist = min_dist
                best_pair = (halogen_atom, carbon_atom)

        halogen_atom, carbon_atom = best_pair

    elif pairs_without_carbon:
        # Fallback: consider halogens without carbon, but we can't optimize without carbon
        # (need carbon to define C-X bond axis for optimization)
        return best_atoms

    else:
        # No valid pairs with halogen and carbon
        return best_atoms

    # Find closest oxygen to halogen
    min_dist = float('inf')
    target_oxygen = None
    for oxygen in acceptor_oxygens:
        dist = np.sqrt(
            (halogen_atom['x'] - oxygen['x'])**2 +
            (halogen_atom['y'] - oxygen['y'])**2 +
            (halogen_atom['z'] - oxygen['z'])**2
        )
        if dist < min_dist:
            min_dist = dist
            target_oxygen = oxygen

    if not target_oxygen:
        return best_atoms

    # Define optimization axes
    # Vector from halogen to carbon (C-X bond direction)
    vec_hc = np.array([
        carbon_atom['x'] - halogen_atom['x'],
        carbon_atom['y'] - halogen_atom['y'],
        carbon_atom['z'] - halogen_atom['z']
    ])

    # Vector from halogen to oxygen (X···O interaction)
    vec_ho = np.array([
        target_oxygen['x'] - halogen_atom['x'],
        target_oxygen['y'] - halogen_atom['y'],
        target_oxygen['z'] - halogen_atom['z']
    ])

    # Normalize vectors
    norm_hc = np.linalg.norm(vec_hc)
    norm_ho = np.linalg.norm(vec_ho)

    if norm_hc > 0 and norm_ho > 0:
        vec_hc_unit = vec_hc / norm_hc
        vec_ho_unit = vec_ho / norm_ho

        # Compute two perpendicular vectors to vec_ho_unit for translational degrees of freedom
        # Choose an arbitrary vector not parallel to vec_ho_unit
        if abs(vec_ho_unit[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(vec_ho_unit, arbitrary)
        perp1_norm = np.linalg.norm(perp1)
        if perp1_norm > 0:
            perp1 = perp1 / perp1_norm
        else:
            # If cross product is zero (shouldn't happen with our arbitrary choice), try another
            perp1 = np.cross(vec_ho_unit, np.array([0.0, 0.0, 1.0]))
            perp1_norm = np.linalg.norm(perp1)
            if perp1_norm > 0:
                perp1 = perp1 / perp1_norm
            else:
                # Fallback: set to arbitrary perpendicular vectors
                perp1 = np.array([1.0, 0.0, 0.0])
                perp2 = np.array([0.0, 1.0, 0.0])
        perp2 = np.cross(vec_ho_unit, perp1)  # Already unit if inputs are unit and perpendicular

        # Define optimization parameters
        # Existing degrees of freedom
        translation_steps = 7  # -0.4, -0.2, 0, +0.2, +0.4 Å along X···O
        rotation_steps = 7     # -15°, -10°, -5°, 0, +5°, +10°, +15° around axis perp to C-X and X···O
        max_translation = 0.4  # Å
        max_rotation = np.radians(8)  # radians

        # New degrees of freedom for avoiding repulsive clashes
        trans_perp_steps = 3  # finer grid along perp1 and perp2
        rot_ho_steps = 3  # finer grid around X-O axis
        max_trans_perp = 0.3  # A - increased to allow larger lateral shifts
        max_rot_ho = np.radians(10)  # radians - increased for clash avoidance

        # Grid search over all degrees of freedom
        for t_idx in range(translation_steps):  # translation along X···O
            # Translation factor: -1 to +1
            t_factor = -1.0 + (2.0 * t_idx / (translation_steps - 1)) if translation_steps > 1 else 0.0
            translation_ho = vec_ho_unit * (max_translation * t_factor)

            for r_idx in range(rotation_steps):  # rotation around axis perp to C-X and X···O
                # Rotation factor: -1 to +1
                r_factor = -1.0 + (2.0 * r_idx / (rotation_steps - 1)) if rotation_steps > 1 else 0.0
                rotation_angle_perp = max_rotation * r_factor

                for tp1_idx in range(trans_perp_steps):  # translation along perp1
                    tp1_factor = -1.0 + (2.0 * tp1_idx / (trans_perp_steps - 1)) if trans_perp_steps > 1 else 0.0
                    translation_perp1 = perp1 * (max_trans_perp * tp1_factor)

                    for tp2_idx in range(trans_perp_steps):  # translation along perp2
                        tp2_factor = -1.0 + (2.0 * tp2_idx / (trans_perp_steps - 1)) if trans_perp_steps > 1 else 0.0
                        translation_perp2 = perp2 * (max_trans_perp * tp2_factor)

                        for rho_idx in range(rot_ho_steps):  # rotation around X···O axis
                            rho_factor = -1.0 + (2.0 * rho_idx / (rot_ho_steps - 1)) if rot_ho_steps > 1 else 0.0
                            rotation_angle_ho = max_rot_ho * rho_factor

                            # Apply transformation: rotate, then translate
                            test_atoms = copy.deepcopy(ligand_atoms)

                            # Apply rotations
                            # First: rotation around axis perp to C-X and X···O (existing)
                            if rotation_angle_perp != 0:
                                rotation_axis_perp = np.cross(vec_hc_unit, vec_ho_unit)  # perp to both
                                axis_norm = np.linalg.norm(rotation_axis_perp)
                                if axis_norm > 0:
                                    rotation_axis_perp = rotation_axis_perp / axis_norm
                                    # Apply Rodrigues' rotation formula
                                    cos_theta = np.cos(rotation_angle_perp)
                                    sin_theta = np.sin(rotation_angle_perp)

                                    for atom in test_atoms:
                                        # Vector from halogen to atom
                                        vec_ha = np.array([
                                            atom['x'] - halogen_atom['x'],
                                            atom['y'] - halogen_atom['y'],
                                            atom['z'] - halogen_atom['z']
                                        ])

                                        # Rodrigues' rotation formula
                                        vec_ha_rotated = (
                                            vec_ha * cos_theta +
                                            np.cross(rotation_axis_perp, vec_ha) * sin_theta +
                                            rotation_axis_perp * np.dot(rotation_axis_perp, vec_ha) * (1 - cos_theta)
                                        )

                                        # Update atom position
                                        atom['x'] = halogen_atom['x'] + vec_ha_rotated[0]
                                        atom['y'] = halogen_atom['y'] + vec_ha_rotated[1]
                                        atom['z'] = halogen_atom['z'] + vec_ha_rotated[2]

                            # Second: rotation around X···O axis (new)
                            if rotation_angle_ho != 0:
                                rotation_axis_ho = vec_ho_unit  # rotation around X···O axis
                                # Already normalized
                                # Apply Rodrigues' rotation formula
                                cos_theta = np.cos(rotation_angle_ho)
                                sin_theta = np.sin(rotation_angle_ho)

                                for atom in test_atoms:
                                    # Vector from halogen to atom
                                    vec_ha = np.array([
                                        atom['x'] - halogen_atom['x'],
                                        atom['y'] - halogen_atom['y'],
                                        atom['z'] - halogen_atom['z']
                                    ])

                                    # Rodrigues' rotation formula
                                    vec_ha_rotated = (
                                        vec_ha * cos_theta +
                                        np.cross(rotation_axis_ho, vec_ha) * sin_theta +
                                        rotation_axis_ho * np.dot(rotation_axis_ho, vec_ha) * (1 - cos_theta)
                                        )

                                    # Update atom position
                                    atom['x'] = halogen_atom['x'] + vec_ha_rotated[0]
                                    atom['y'] = halogen_atom['y'] + vec_ha_rotated[1]
                                    atom['z'] = halogen_atom['z'] + vec_ha_rotated[2]

                            # Apply translations
                            # First: translation along X···O (existing)
                            if np.linalg.norm(translation_ho) > 0:
                                for atom in test_atoms:
                                    atom['x'] += translation_ho[0]
                                    atom['y'] += translation_ho[1]
                                    atom['z'] += translation_ho[2]

                            # Second: translation along perp1 (new)
                            if np.linalg.norm(translation_perp1) > 0:
                                for atom in test_atoms:
                                    atom['x'] += translation_perp1[0]
                                    atom['y'] += translation_perp1[1]
                                    atom['z'] += translation_perp1[2]

                            # Third: translation along perp2 (new)
                            if np.linalg.norm(translation_perp2) > 0:
                                for atom in test_atoms:
                                    atom['x'] += translation_perp2[0]
                                    atom['y'] += translation_perp2[1]
                                    atom['z'] += translation_perp2[2]

                            # Calculate energy for this pose
                            # Note: _calculate_pairwise_energy would need to be imported from scoring module
                            energy = 0.0  # Placeholder - will be updated when imported

                            # Update best if this is better (more negative)
                            if energy < best_energy:
                                best_energy = energy
                                best_atoms = copy.deepcopy(test_atoms)

    return best_atoms