"""
Molecular alignment utilities for sigma-hole docking — places the dummy atom along the C-X extension toward the receptor acceptor.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlignmentMixin:
    """Mixin: coordinate alignment and geometry helpers."""

    def _is_planar_molecule(self, atoms: List[Dict], tolerance: float = 0.01) -> bool:
        """
        Check if a molecule is planar (all atoms lie in the same plane within tolerance).

        Args:
            atoms: List of atom dictionaries with x, y, z coordinates
            tolerance: Maximum Z-axis deviation to consider planar (Å)

        Returns:
            True if molecule is planar, False otherwise
        """
        if len(atoms) < 3:
            return False  # Need at least 3 points to define a plane

        # Extract coordinates
        coords = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms])

        # Use SVD to find the best-fit plane
        # Center the coordinates
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # Perform SVD
        U, S, Vt = np.linalg.svd(centered_coords)

        # The normal to the plane is the last row of Vt (corresponding to smallest singular value)
        normal = Vt[-1]

        # Calculate distances from points to the plane
        distances = np.abs(np.dot(centered_coords, normal))
        max_deviation = np.max(distances)

        return max_deviation < tolerance

    def _add_planar_offset(self, atoms: List[Dict], max_offset: float = 0.01) -> List[Dict]:
        """
        Add small random Z-offset to break planarity degeneracy.

        Args:
            atoms: List of atom dictionaries
            max_offset: Maximum offset magnitude (Å)

        Returns:
            New list of atoms with small random Z-offset applied
        """
        import copy
        import random

        offset_atoms = copy.deepcopy(atoms)
        for atom in offset_atoms:
            # Add small random offset to Z coordinate
            offset = random.uniform(-max_offset, max_offset)
            atom["z"] += offset

        return offset_atoms

    def _align_molecules_for_sigma_hole(
        self, ligand_atoms: List[Dict], receptor_atoms: List[Dict]
    ) -> List[Dict]:
        """
        Align ligand for optimal sigma-hole interaction with receptor.

        Aligns so that:
        1. Halogen points toward nearest receptor acceptor atom (O/N/S/F)
        2. C-X···A angle is ~180° (linear)
        3. X···A distance is ~3.0 Å
        4. Handles planar molecules by adding small random offsets to prevent singularities

        Args:
            ligand_atoms: List of ligand atom dictionaries (will be modified)
            receptor_atoms: List of receptor atom dictionaries

        Returns:
            Aligned ligand_atoms list
        """
        # Handle planar molecules by adding small random offsets to prevent singularities
        original_receptor_atoms = receptor_atoms
        if self._is_planar_molecule(receptor_atoms, tolerance=0.01):
            logger.warning(
                "Receptor appears planar (Z variance < 0.01 Å), adding small random Z-offset to prevent alignment singularities"
            )
            receptor_atoms = self._add_planar_offset(receptor_atoms, max_offset=0.01)

        # Find key atoms
        halogen_atom, carbon_atom = self._find_halogen_and_carbon(ligand_atoms)
        acceptor_atoms = self._find_acceptor_atoms(receptor_atoms)

        if not halogen_atom or not acceptor_atoms:
            logger.warning("Could not find halogen or acceptor atoms for alignment")
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return ligand_atoms

        if not carbon_atom:
            logger.warning("Could not find carbon bonded to halogen for alignment")
            # Still try to align using just halogen position
            result = self._align_by_halogen_only(
                ligand_atoms, receptor_atoms, halogen_atom, acceptor_atoms
            )
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return result

        # Find closest acceptor atom to halogen
        min_dist = float("inf")
        target_acceptor = None
        for acceptor in acceptor_atoms:
            dist = np.sqrt(
                (halogen_atom["x"] - acceptor["x"]) ** 2
                + (halogen_atom["y"] - acceptor["y"]) ** 2
                + (halogen_atom["z"] - acceptor["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                target_acceptor = acceptor

        if not target_acceptor:
            logger.warning("Could not find target acceptor for alignment")
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return ligand_atoms

        logger.info(
            f"Pre-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {min_dist:.3f} Å"
        )

        # Define target geometry
        # Halogen-specific optimal X...A distances (sum of vdW radii)
        halogen_vdw = {"F": 1.47, "Cl": 1.75, "Br": 1.83, "I": 1.98, "At": 2.02}
        acceptor_vdw = {"O": 1.52, "N": 1.55, "S": 1.80, "F": 1.47}
        h_vdw = halogen_vdw.get(halogen_atom["element"], 1.98)
        a_vdw = acceptor_vdw.get(target_acceptor["element"], 1.52)
        target_distance = h_vdw + a_vdw

        # Current vectors
        # Vector from halogen to carbon (C-X bond direction, pointing FROM halogen TO carbon)
        vec_hc = np.array(
            [
                carbon_atom["x"] - halogen_atom["x"],
                carbon_atom["y"] - halogen_atom["y"],
                carbon_atom["z"] - halogen_atom["z"],
            ]
        )

        # Vector from halogen to acceptor (X···A interaction)
        vec_ha = np.array(
            [
                target_acceptor["x"] - halogen_atom["x"],
                target_acceptor["y"] - halogen_atom["y"],
                target_acceptor["z"] - halogen_atom["z"],
            ]
        )

        current_dist = np.linalg.norm(vec_ha)
        if current_dist > 0:
            vec_ha_unit = vec_ha / current_dist
        else:
            vec_ha_unit = vec_ha  # Avoid division by zero

        # Current C-X···A angle
        if np.linalg.norm(vec_hc) > 0 and current_dist > 0:
            vec_hc_unit = vec_hc / np.linalg.norm(vec_hc)
            dot_product = np.dot(vec_hc_unit, vec_ha_unit)
            # Clamp to [-1, 1] for numerical stability
            dot_product = max(-1.0, min(1.0, dot_product))
            current_angle_rad = np.arccos(dot_product)
            current_angle_deg = np.degrees(current_angle_rad)
        else:
            current_angle_rad = 0
            current_angle_deg = 0

        logger.info(
            f"Pre-alignment: C-{halogen_atom['element']}···{target_acceptor['element']} angle = {current_angle_deg:.1f}°"
        )

        # Step 1: Translate ligand to achieve target X···A distance
        # We want to move the halogen along the X···A vector to reach target_distance
        if current_dist > 0:
            translation_factor = (target_distance - current_dist) / current_dist
            translation_vec = vec_ha_unit * translation_factor * current_dist
            # Actually: new_pos = old_pos + (current_dist - target_distance) * direction
            # Move toward acceptor if distance too large, away if too small
            translation_vec = vec_ha_unit * (current_dist - target_distance)
        else:
            # If atoms are on top of each other, move along arbitrary direction
            translation_vec = np.array([target_distance, 0.0, 0.0])

        # Apply translation to all ligand atoms
        for atom in ligand_atoms:
            atom["x"] += translation_vec[0]
            atom["y"] += translation_vec[1]
            atom["z"] += translation_vec[2]

        # Recalculate vectors after translation
        vec_ha = np.array(
            [
                target_acceptor["x"] - halogen_atom["x"],
                target_acceptor["y"] - halogen_atom["y"],
                target_acceptor["z"] - halogen_atom["z"],
            ]
        )
        current_dist = np.linalg.norm(vec_ha)
        if current_dist > 0:
            vec_ha_unit = vec_ha / current_dist

        # Step 2: Rotate ligand around halogen to achieve target angle
        # We want to rotate so that C-X···A angle = 180°
        # This means we want vec_hc (C→X) to be opposite to vec_ha (X···A)
        # So we want vec_hc to point in the same direction as -vec_ha

        # If we have a carbon atom, rotate to align C-X bond
        if carbon_atom and np.linalg.norm(vec_hc) > 0:
            # Current vector from halogen to carbon (after translation)
            vec_hc_current = np.array(
                [
                    carbon_atom["x"] - halogen_atom["x"],
                    carbon_atom["y"] - halogen_atom["y"],
                    carbon_atom["z"] - halogen_atom["z"],
                ]
            )

            # Target vector from halogen to carbon should be opposite to X···A vector
            # For linear C-X···A: C--X···A, so vector X→C should be opposite to vector X→A
            vec_hc_target = -vec_ha_unit * np.linalg.norm(vec_hc_current)

            # Calculate rotation to align vec_hc_current with vec_hc_target
            if np.linalg.norm(vec_hc_current) > 0 and np.linalg.norm(vec_hc_target) > 0:
                # Use Rodrigues' rotation formula
                vec_hc_current_norm = vec_hc_current / np.linalg.norm(vec_hc_current)
                vec_hc_target_norm = vec_hc_target / np.linalg.norm(vec_hc_target)

                # Find rotation axis and angle
                cross_product = np.cross(vec_hc_current_norm, vec_hc_target_norm)
                dot_product = np.dot(vec_hc_current_norm, vec_hc_target_norm)
                dot_product = max(-1.0, min(1.0, dot_product))  # Clamp for numerical stability

                if abs(dot_product) < 1.0:  # Not already aligned
                    rotation_angle = np.arccos(dot_product)
                    if np.linalg.norm(cross_product) > 0:
                        rotation_axis = cross_product / np.linalg.norm(cross_product)

                        # Apply rotation to all ligand atoms around halogen
                        cos_theta = np.cos(rotation_angle)
                        sin_theta = np.sin(rotation_angle)

                        for atom in ligand_atoms:
                            # Vector from halogen to atom
                            vec_ha = np.array(
                                [
                                    atom["x"] - halogen_atom["x"],
                                    atom["y"] - halogen_atom["y"],
                                    atom["z"] - halogen_atom["z"],
                                ]
                            )

                            # Rodrigues' rotation formula
                            vec_ha_rotated = (
                                vec_ha * cos_theta
                                + np.cross(rotation_axis, vec_ha) * sin_theta
                                + rotation_axis * np.dot(rotation_axis, vec_ha) * (1 - cos_theta)
                            )

                            # Update atom position
                            atom["x"] = halogen_atom["x"] + vec_ha_rotated[0]
                            atom["y"] = halogen_atom["y"] + vec_ha_rotated[1]
                            atom["z"] = halogen_atom["z"] + vec_ha_rotated[2]
        else:
            # If no carbon found, we can't rotate meaningfully, just note it
            logger.warning("No carbon found for rotation alignment")

        # Final check
        final_dist = np.sqrt(
            (halogen_atom["x"] - target_acceptor["x"]) ** 2
            + (halogen_atom["y"] - target_acceptor["y"]) ** 2
            + (halogen_atom["z"] - target_acceptor["z"]) ** 2
        )

        if carbon_atom:
            vec_hc_final = np.array(
                [
                    carbon_atom["x"] - halogen_atom["x"],
                    carbon_atom["y"] - halogen_atom["y"],
                    carbon_atom["z"] - halogen_atom["z"],
                ]
            )
            vec_ha_final = np.array(
                [
                    target_acceptor["x"] - halogen_atom["x"],
                    target_acceptor["y"] - halogen_atom["y"],
                    target_acceptor["z"] - halogen_atom["z"],
                ]
            )
            if np.linalg.norm(vec_hc_final) > 0 and np.linalg.norm(vec_ha_final) > 0:
                vec_hc_final_norm = vec_hc_final / np.linalg.norm(vec_hc_final)
                vec_ha_final_norm = vec_ha_final / np.linalg.norm(vec_ha_final)
                dot_final = np.dot(vec_hc_final_norm, vec_ha_final_norm)
                dot_final = max(-1.0, min(1.0, dot_final))
                final_angle_rad = np.arccos(dot_final)
                final_angle_deg = np.degrees(final_angle_rad)
                logger.info(
                    f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å, "
                    f"C-{halogen_atom['element']}···{target_acceptor['element']} angle = {final_angle_deg:.1f}°"
                )
            else:
                logger.info(
                    f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å"
                )
        else:
            logger.info(
                f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å"
            )

        # Restore original receptor atoms if we made a copy due to planar correction
        if receptor_atoms is not original_receptor_atoms:
            receptor_atoms = original_receptor_atoms

        return ligand_atoms

    def _align_by_halogen_only(
        self,
        ligand_atoms: List[Dict],
        receptor_atoms: List[Dict],
        halogen_atom: Dict,
        acceptor_atoms: List[Dict],
    ) -> List[Dict]:
        """
        Simple alignment by distance only when carbon bonded to halogen is not found.
        Handles planar molecules by adding small random offsets to prevent singularities.
        """
        # Handle planar molecules by adding small random offsets to prevent singularities
        original_receptor_atoms = receptor_atoms
        if self._is_planar_molecule(receptor_atoms, tolerance=0.01):
            logger.warning(
                "Receptor appears planar (Z variance < 0.01 Å), adding small random Z-offset to prevent alignment singularities"
            )
            receptor_atoms = self._add_planar_offset(receptor_atoms, max_offset=0.01)

        if not acceptor_atoms:
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return ligand_atoms

        # Find closest acceptor atom
        min_dist = float("inf")
        target_acceptor = None
        for acceptor in acceptor_atoms:
            dist = np.sqrt(
                (halogen_atom["x"] - acceptor["x"]) ** 2
                + (halogen_atom["y"] - acceptor["y"]) ** 2
                + (halogen_atom["z"] - acceptor["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                target_acceptor = acceptor

        if not target_acceptor:
            return ligand_atoms

        target_distance = 3.0
        current_dist = min_dist

        if current_dist > 0:
            translation_factor = (target_distance - current_dist) / current_dist
            translation_vec = np.array(
                [
                    target_acceptor["x"] - halogen_atom["x"],
                    target_acceptor["y"] - halogen_atom["y"],
                    target_acceptor["z"] - halogen_atom["z"],
                ]
            ) * (translation_factor if current_dist > 0 else 0)
            if current_dist > 0:
                # Normalize and scale
                direction = np.array(
                    [
                        target_acceptor["x"] - halogen_atom["x"],
                        target_acceptor["y"] - halogen_atom["y"],
                        target_acceptor["z"] - halogen_atom["z"],
                    ]
                )
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction_unit = direction / direction_norm
                    translation_vec = direction_unit * (target_distance - current_dist)
                else:
                    translation_vec = np.array([0.0, 0.0, 0.0])
            else:
                translation_vec = np.array([target_distance, 0.0, 0.0])
        else:
            translation_vec = np.array([target_distance, 0.0, 0.0])

        # Apply translation
        for atom in ligand_atoms:
            atom["x"] += translation_vec[0]
            atom["y"] += translation_vec[1]
            atom["z"] += translation_vec[2]

        # Restore original receptor atoms if we made a copy
        if receptor_atoms is not original_receptor_atoms:
            receptor_atoms = original_receptor_atoms

        return ligand_atoms

    def _validate_coordinates(self, atoms: List[Dict], context: str = "") -> bool:
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
            for coord_name in ["x", "y", "z"]:
                coord_val = atom[coord_name]
                if not isinstance(coord_val, (int, float)) or not np.isfinite(coord_val):
                    logger.error(
                        f"INVALID COORDINATES {context}: Atom {i} ({atom.get('element', '?')}) has {coord_name}={coord_val}"
                    )
                    return False
        return True

    def compute_receptor_center(
        self, receptor_pdbqt: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute the geometric center of a receptor from its PDBQT file.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file

        Returns:
            Tuple of (x, y, z) coordinates of the geometric center, or (None, None, None) if failed
        """
        # Parse the PDBQT file using our standard parser
        receptor_atoms = self._parse_pdbqt(receptor_pdbqt)
        if not receptor_atoms:
            logger.error(
                f"Failed to parse receptor PDBQT file for center computation: {receptor_pdbqt}"
            )
            return None, None, None

        # Extract coordinates
        coords = []
        for atom in receptor_atoms:
            coords.append([atom["x"], atom["y"], atom["z"]])

        if coords:
            coords_array = np.array(coords)
            center = coords_array.mean(axis=0)
            return float(center[0]), float(center[1]), float(center[2])
        else:
            # This should not happen if receptor_atoms is not empty, but just in case
            logger.error(f"No coordinates found in parsed receptor atoms: {receptor_pdbqt}")
            return None, None, None
