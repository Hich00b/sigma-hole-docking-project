"""
Sigma Hole Docking Engine

Handles docking and scoring calculations that properly evaluate
dummy atom electrostatics for sigma-hole interactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import subprocess
import os
import tempfile
import math

logger = logging.getLogger(__name__)

class SigmaHoleDockingEngine:
    """
    Docking engine for sigma-hole interactions.

    Implements multiple scoring approaches:
    1. Physics-based scoring (Lennard-Jones + Coulomb) - reliable for dummy atoms
    2. Smina with custom scoring - if available
    3. Corrected Vina/Smina AD4 scoring - attempts to fix charge reading issues
    4. GB/SA or other implicit solvation models - if available
    """

    def __init__(self, use_physics_fallback: bool = True, dielectric_coeff: float = 0.0, charge_scale: float = 1.0):
        """
        Initialize the docking engine.

        Args:
            use_physics_fallback: Whether to use physics-based scoring as fallback
            dielectric_coeff: Coefficient for constant dielectric model.
                Coulomb uses epsilon_r = max(dielectric_coeff, 1.0), which models
                dielectric screening. For gas-phase calculations (epsilon_r=1),
                set dielectric_coeff=0.0. For solvent screening, use values > 0.
                Typical solvent range: 2-4.
        """
        self.use_physics_fallback = use_physics_fallback
        self.k_coulomb = 332.06  # kcal·Å/(mol·e²)
        self.dielectric_coeff = dielectric_coeff
        self.charge_scale = charge_scale

        # Default Lennard-Jones parameters for common atom pairs
        # Format: (epsilon, sigma) where epsilon is well depth (kcal/mol), sigma is distance (Å)
        self.lj_params = {
            # (atom1, atom2): (epsilon, sigma)
            ('O', 'H'): (0.075, 2.0),   # Oxygen-hydrogen (for dummy H)
            ('O', 'C'): (0.10, 3.5),   # Oxygen-carbon
            ('O', 'O'): (0.05, 3.0),   # Oxygen-oxygen
            ('C', 'H'): (0.025, 2.5),   # Carbon-hydrogen
            ('C', 'C'): (0.05, 3.4),   # Carbon-carbon - FIXED: was 4.0, too high causing repulsive energies
            # Halogen-specific parameters (approximate)
            # Halogen-oxygen: real vdW attraction for steric repulsion
            ('O', 'F'): (0.05, 2.8),
            ('O', 'Cl'): (0.10, 3.1),
            ('O', 'Br'): (0.15, 3.2),
            ('O', 'I'): (0.20, 3.3),
            # Halogen-carbon
            ('C', 'F'): (0.05, 2.9),
            ('C', 'Cl'): (0.10, 3.2),
            ('C', 'Br'): (0.15, 3.3),
            ('C', 'I'): (0.20, 3.5),
            # Halogen-nitrogen
            ('N', 'F'): (0.05, 2.8),
            ('N', 'Cl'): (0.10, 3.1),
            ('N', 'Br'): (0.15, 3.2),
            ('N', 'I'): (0.20, 3.3),
            # Halogen-sulfur
            ('S', 'F'): (0.05, 3.0),
            ('S', 'Cl'): (0.10, 3.3),
            ('S', 'Br'): (0.15, 3.4),
            ('S', 'I'): (0.20, 3.6),
        }


    def _compute_cx_acceptor_angle(self, halogen_atom, acceptor_atom, ligand_atoms):
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

    def _halogen_acceptor_charge_scale(self, angle_deg):
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

    def _bonded_carbon_charge_scale(self, angle_deg):
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

    def _dummy_acceptor_charge_scale(self, rec_element):
        """Scale dummy atom Coulomb by receptor atom type.
        The sigma-hole only attracts electronegative acceptors (O, N, S, F).
        Other receptor atoms should not interact with the dummy."""
        if rec_element in ['O', 'N', 'S', 'F']:
            return 1.0
        else:
            return 0.0

    def _find_bonded_carbon(self, halogen_atom, ligand_atoms):
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

    def _get_lj_parameters(self, atom1: str, atom2: str) -> Tuple[float, float]:
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

        if key1 in self.lj_params:
            return self.lj_params[key1]
        elif key2 in self.lj_params:
            return self.lj_params[key2]

        # Use mixing rules: epsilon = sqrt(eps1*eps2), sigma = (sigma1+sigma2)/2
        # Default parameters for common atoms
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

    def _find_halogen_and_carbon(self, ligand_atoms: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Find halogen atom and the carbon bonded to it in ligand.

        Returns:
            Tuple of (halogen_atom, carbon_atom) where:
            - halogen_atom: Dict with halogen info (F, Cl, Br, I)
            - carbon_atom: Dict of carbon bonded to halogen, or None if not found
        """
        halogen_atom = None
        carbon_atoms = [atom for atom in ligand_atoms if atom['element'] == 'C']

        # Find halogen
        for atom in ligand_atoms:
            if atom['element'] in ['F', 'Cl', 'Br', 'I', 'At']:
                halogen_atom = atom
                break

        if not halogen_atom:
            return None, None

        # Find closest carbon to halogen (assumed to be bonded)
        if carbon_atoms:
            min_dist = float('inf')
            closest_carbon = None
            for carbon in carbon_atoms:
                dist = np.sqrt(
                    (halogen_atom['x'] - carbon['x'])**2 +
                    (halogen_atom['y'] - carbon['y'])**2 +
                    (halogen_atom['z'] - carbon['z'])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_carbon = carbon
            return halogen_atom, closest_carbon

        return halogen_atom, None

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
        coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])

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
            atom['z'] += offset

        return offset_atoms

    def _find_acceptor_atoms(self, receptor_atoms: List[Dict]) -> List[Dict]:
        """
        Find electronegative atoms in receptor that can act as sigma-hole acceptors.
        Priority order: O > N > S > F (based on electronegativity and common sigma-hole interactions)

        Args:
            receptor_atoms: List of receptor atom dictionaries

        Returns:
            List of acceptor atom dictionaries (prioritized by electronegativity)
        """
        # Define acceptor elements in priority order (highest electronegativity first)
        acceptor_elements = ['O', 'N', 'S', 'F']

        acceptor_atoms = []
        for element in acceptor_elements:
            element_atoms = [atom for atom in receptor_atoms if atom['element'] == element]
            if element_atoms:
                acceptor_atoms.extend(element_atoms)
                # Log which acceptor type was found
                logger.info(f"Found {len(element_atoms)} {element} acceptor atom(s)")
                # Continue to collect all acceptor types (O, N, S, F)

        if not acceptor_atoms:
            logger.warning("No acceptor atoms (O/N/S/F) found in receptor")
            # Fallback to oxygen-only for backward compatibility
            acceptor_atoms = [atom for atom in receptor_atoms if atom['element'] == 'O']

        return acceptor_atoms

    def _align_molecules_for_sigma_hole(self,
                                      ligand_atoms: List[Dict],
                                      receptor_atoms: List[Dict]) -> List[Dict]:
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
            logger.warning("Receptor appears planar (Z variance < 0.01 Å), adding small random Z-offset to prevent alignment singularities")
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
            result = self._align_by_halogen_only(ligand_atoms, receptor_atoms, halogen_atom, acceptor_atoms)
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return result

        # Find closest acceptor atom to halogen
        min_dist = float('inf')
        target_acceptor = None
        for acceptor in acceptor_atoms:
            dist = np.sqrt(
                (halogen_atom['x'] - acceptor['x'])**2 +
                (halogen_atom['y'] - acceptor['y'])**2 +
                (halogen_atom['z'] - acceptor['z'])**2
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

        logger.info(f"Pre-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {min_dist:.3f} Å")

        # Define target geometry
        # Halogen-specific optimal X...A distances (sum of vdW radii)
        halogen_vdw = {'F': 1.47, 'Cl': 1.75, 'Br': 1.83, 'I': 1.98, 'At': 2.02}
        acceptor_vdw = {'O': 1.52, 'N': 1.55, 'S': 1.80, 'F': 1.47}
        h_vdw = halogen_vdw.get(halogen_atom['element'], 1.98)
        a_vdw = acceptor_vdw.get(target_acceptor['element'], 1.52)
        target_distance = h_vdw + a_vdw
        target_angle_rad = np.pi  # 180° in radians

        # Current vectors
        # Vector from halogen to carbon (C-X bond direction, pointing FROM halogen TO carbon)
        vec_hc = np.array([
            carbon_atom['x'] - halogen_atom['x'],
            carbon_atom['y'] - halogen_atom['y'],
            carbon_atom['z'] - halogen_atom['z']
        ])

        # Vector from halogen to acceptor (X···A interaction)
        vec_ha = np.array([
            target_acceptor['x'] - halogen_atom['x'],
            target_acceptor['y'] - halogen_atom['y'],
            target_acceptor['z'] - halogen_atom['z']
        ])

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

        logger.info(f"Pre-alignment: C-{halogen_atom['element']}···{target_acceptor['element']} angle = {current_angle_deg:.1f}°")

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
            atom['x'] += translation_vec[0]
            atom['y'] += translation_vec[1]
            atom['z'] += translation_vec[2]

        # Recalculate vectors after translation
        vec_ha = np.array([
            target_acceptor['x'] - halogen_atom['x'],
            target_acceptor['y'] - halogen_atom['y'],
            target_acceptor['z'] - halogen_atom['z']
        ])
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
            vec_hc_current = np.array([
                carbon_atom['x'] - halogen_atom['x'],
                carbon_atom['y'] - halogen_atom['y'],
                carbon_atom['z'] - halogen_atom['z']
            ])

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
                            vec_ha = np.array([
                                atom['x'] - halogen_atom['x'],
                                atom['y'] - halogen_atom['y'],
                                atom['z'] - halogen_atom['z']
                            ])

                            # Rodrigues' rotation formula
                            vec_ha_rotated = (
                                vec_ha * cos_theta +
                                np.cross(rotation_axis, vec_ha) * sin_theta +
                                rotation_axis * np.dot(rotation_axis, vec_ha) * (1 - cos_theta)
                            )

                            # Update atom position
                            atom['x'] = halogen_atom['x'] + vec_ha_rotated[0]
                            atom['y'] = halogen_atom['y'] + vec_ha_rotated[1]
                            atom['z'] = halogen_atom['z'] + vec_ha_rotated[2]
        else:
            # If no carbon found, we can't rotate meaningfully, just note it
            logger.warning("No carbon found for rotation alignment")

        # Final check
        final_dist = np.sqrt(
            (halogen_atom['x'] - target_acceptor['x'])**2 +
            (halogen_atom['y'] - target_acceptor['y'])**2 +
            (halogen_atom['z'] - target_acceptor['z'])**2
        )

        if carbon_atom:
            vec_hc_final = np.array([
                carbon_atom['x'] - halogen_atom['x'],
                carbon_atom['y'] - halogen_atom['y'],
                carbon_atom['z'] - halogen_atom['z']
            ])
            vec_ha_final = np.array([
                target_acceptor['x'] - halogen_atom['x'],
                target_acceptor['y'] - halogen_atom['y'],
                target_acceptor['z'] - halogen_atom['z']
            ])
            if np.linalg.norm(vec_hc_final) > 0 and np.linalg.norm(vec_ha_final) > 0:
                vec_hc_final_norm = vec_hc_final / np.linalg.norm(vec_hc_final)
                vec_ha_final_norm = vec_ha_final / np.linalg.norm(vec_ha_final)
                dot_final = np.dot(vec_hc_final_norm, vec_ha_final_norm)
                dot_final = max(-1.0, min(1.0, dot_final))
                final_angle_rad = np.arccos(dot_final)
                final_angle_deg = np.degrees(final_angle_rad)
                logger.info(f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å, "
                           f"C-{halogen_atom['element']}···{target_acceptor['element']} angle = {final_angle_deg:.1f}°")
            else:
                logger.info(f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å")
        else:
            logger.info(f"Post-alignment: {halogen_atom['element']}···{target_acceptor['element']} distance = {final_dist:.3f} Å")

        # Restore original receptor atoms if we made a copy due to planar correction
        if receptor_atoms is not original_receptor_atoms:
            receptor_atoms = original_receptor_atoms

        return ligand_atoms

    def _align_by_halogen_only(self,
                              ligand_atoms: List[Dict],
                              receptor_atoms: List[Dict],
                              halogen_atom: Dict,
                              acceptor_atoms: List[Dict]) -> List[Dict]:
        """
        Simple alignment by distance only when carbon bonded to halogen is not found.
        Handles planar molecules by adding small random offsets to prevent singularities.
        """
        # Handle planar molecules by adding small random offsets to prevent singularities
        original_receptor_atoms = receptor_atoms
        if self._is_planar_molecule(receptor_atoms, tolerance=0.01):
            logger.warning("Receptor appears planar (Z variance < 0.01 Å), adding small random Z-offset to prevent alignment singularities")
            receptor_atoms = self._add_planar_offset(receptor_atoms, max_offset=0.01)

        if not acceptor_atoms:
            # Restore original receptor atoms if we made a copy
            if receptor_atoms is not original_receptor_atoms:
                receptor_atoms = original_receptor_atoms
            return ligand_atoms

        # Find closest acceptor atom
        min_dist = float('inf')
        target_acceptor = None
        for acceptor in acceptor_atoms:
            dist = np.sqrt(
                (halogen_atom['x'] - acceptor['x'])**2 +
                (halogen_atom['y'] - acceptor['y'])**2 +
                (halogen_atom['z'] - acceptor['z'])**2
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
            translation_vec = np.array([
                target_acceptor['x'] - halogen_atom['x'],
                target_acceptor['y'] - halogen_atom['y'],
                target_acceptor['z'] - halogen_atom['z']
            ]) * (translation_factor if current_dist > 0 else 0)
            if current_dist > 0:
                # Normalize and scale
                direction = np.array([
                    target_acceptor['x'] - halogen_atom['x'],
                    target_acceptor['y'] - halogen_atom['y'],
                    target_acceptor['z'] - halogen_atom['z']
                ])
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
            atom['x'] += translation_vec[0]
            atom['y'] += translation_vec[1]
            atom['z'] += translation_vec[2]

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
            for coord_name in ['x', 'y', 'z']:
                coord_val = atom[coord_name]
                if not isinstance(coord_val, (int, float)) or not np.isfinite(coord_val):
                    logger.error(f"INVALID COORDINATES {context}: Atom {i} ({atom.get('element', '?')}) has {coord_name}={coord_val}")
                    return False
        return True

    def _local_optimize_pose(self, ligand_atoms: List[Dict], receptor_atoms: List[Dict]) -> List[Dict]:
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
        import copy
        best_atoms = copy.deepcopy(ligand_atoms)
        best_energy = self._calculate_pairwise_energy(best_atoms, receptor_atoms)

        # Find key atoms for defining the optimization axes
        halogen_atom, carbon_atom = self._find_halogen_and_carbon(best_atoms)
        acceptor_oxygens = self._find_acceptor_atoms(receptor_atoms)

        if not halogen_atom or not acceptor_oxygens or not carbon_atom:
            # Can't optimize without the key atoms, return original
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
                                energy = self._calculate_pairwise_energy(test_atoms, receptor_atoms)

                                # Update best if this is better (more negative)
                                if energy < best_energy:
                                    best_energy = energy
                                    best_atoms = copy.deepcopy(test_atoms)

        return best_atoms

    def _calculate_pairwise_energy(self, ligand_atoms: List[Dict], receptor_atoms: List[Dict]) -> float:
        """
        Calculate pairwise energy between ligand and receptor atoms (helper for optimization).
        Uses the same physics as calculate_physics_score but without alignment/separation.
        Includes directional Coulomb corrections for sigma-hole interactions.
        """
        total_energy = 0.0
        pairs_count = 0

        # Pre-find halogen and bonded carbon for directional Coulomb correction
        _halogen = None
        _bonded_c = None
        for _a in ligand_atoms:
            if _a['element'] in ['F', 'Cl', 'Br', 'I', 'At']:
                _halogen = _a
                break
        if _halogen is not None:
            _bonded_c = self._find_bonded_carbon(_halogen, ligand_atoms)

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

                # Initialize energy variables to avoid UnboundLocalError
                lj_energy = 0.0
                coulomb_energy = 0.0

                # Lennard-Jones (Van der Waals)
                is_lig_dummy = lig_atom.get('is_dummy', False)
                is_rec_dummy = rec_atom.get('is_dummy', False)

                if is_lig_dummy or is_rec_dummy:
                    epsilon, sigma = 0.02, 1.2
                else:
                    epsilon, sigma = self._get_lj_parameters(
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
                    epsilon_r = max(self.dielectric_coeff, 1.0)

                    # Determine charge scale factor for this pair
                    charge_factor = 1.0
                    if lig_atom is _halogen:
                        # Halogen-acceptor: suppress in sigma-hole direction
                        angle = self._compute_cx_acceptor_angle(_halogen, rec_atom, ligand_atoms)
                        charge_factor = self._halogen_acceptor_charge_scale(angle)
                    elif _bonded_c is not None and lig_atom is _bonded_c:
                        # Bonded carbon-acceptor: suppress in sigma-hole direction
                        angle = self._compute_cx_acceptor_angle(_halogen, rec_atom, ligand_atoms)
                        charge_factor = self._bonded_carbon_charge_scale(angle)
                    elif is_lig_dummy:
                        # Dummy atom: only interact with electronegative acceptors
                        charge_factor = self._dummy_acceptor_charge_scale(rec_atom['element'])

                    coulomb_energy = (self.k_coulomb *
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
    def calculate_physics_score(self, ligand_pdbqt: str, receptor_pdbqt: str,
                              cutoff_distance: float = 6.0) -> float:
        """
        Calculate interaction energy using physics-based scoring (LJ + Coulomb).

        This method reliably reads dummy atom charges and positions.
        Molecules are separated along their center-of-mass vector to avoid
        excessive overlap that would lead to unrealistic repulsive energies.
        For sigma-hole interactions, aligns molecules for optimal geometry.

        Args:
            ligand_pdbqt: Path to ligand PDBQT file
            receptor_pdbqt: Path to receptor PDBQT file
            cutoff_distance: Maximum distance for interactions (Å)

        Returns:
            Interaction energy in kcal/mol (negative = favorable)
        """
        try:
            # Parse PDBQT files to get atoms, positions, and charges
            ligand_atoms = self._parse_pdbqt(ligand_pdbqt)
            if not ligand_atoms:
                logger.error(f"Failed to parse ligand PDBQT file: {ligand_pdbqt} - no valid ATOM/HETATM records found")
                return (float('nan'), False)
            logger.info(f"Parsed ligand: {len(ligand_atoms)} atoms from {ligand_pdbqt}")

            receptor_atoms = self._parse_pdbqt(receptor_pdbqt)
            if not receptor_atoms:
                logger.error(f"Failed to parse receptor PDBQT file: {receptor_pdbqt} - no valid ATOM/HETATM records found")
                return (float('nan'), False)
            logger.info(f"Parsed receptor: {len(receptor_atoms)} atoms from {receptor_pdbqt}")

            # Validate that receptor has at least one electronegative atom (O/N/S/F) for sigma-hole interactions
            # This is important because without acceptor atoms, sigma-hole scoring doesn't make sense
            electronegative_elements = {'O', 'N', 'S', 'F'}
            electronegative_count = sum(1 for atom in receptor_atoms if atom['element'] in electronegative_elements)

            if electronegative_count == 0:
                logger.warning(f"No electronegative atoms (O/N/S/F) found in receptor PDBQT: {receptor_pdbqt}")
                logger.warning("Sigma-hole interactions require acceptor atoms (O/N/S/F). Returning NaN.")
                return (float('nan'), False)

            # Align ligand for sigma-hole interaction if halogen and oxygen are present
            logger.info(f"Aligning ligand: {len(ligand_atoms)} ligand atoms, {len(receptor_atoms)} receptor atoms")
            ligand_atoms = self._align_molecules_for_sigma_hole(ligand_atoms, receptor_atoms)
            # Validate coordinates after alignment
            if not self._validate_coordinates(ligand_atoms, "after alignment"):
                logger.error("ALIGNMENT FAILED: Invalid coordinates detected after alignment")
                # Return NaN to indicate failure instead of proceeding with invalid coordinates
                return (float('nan'), False)

            # Perform local optimization to refine the pose and find energy minimum
            logger.info(f"Local optimizing pose: {len(ligand_atoms)} ligand atoms, {len(receptor_atoms)} receptor atoms")
            ligand_atoms = self._local_optimize_pose(ligand_atoms, receptor_atoms)
            # Validate coordinates after optimization
            if not self._validate_coordinates(ligand_atoms, "after local optimization"):
                logger.error("OPTIMIZATION FAILED: Invalid coordinates detected after local optimization")
                # Return NaN to indicate failure instead of proceeding with invalid coordinates
                return (float('nan'), False)

            # Compute geometric centers (for diagnostics only, no separation applied)
            def compute_center(atoms):
                if not atoms:
                    return 0.0, 0.0, 0.0
                sum_x = sum(atom['x'] for atom in atoms)
                sum_y = sum(atom['y'] for atom in atoms)
                sum_z = sum(atom['z'] for atom in atoms)
                count = len(atoms)
                return sum_x / count, sum_y / count, sum_z / count

            lig_center_x, lig_center_y, lig_center_z = compute_center(ligand_atoms)
            rec_center_x, rec_center_y, rec_center_z = compute_center(receptor_atoms)

            # Disabled overlap resolution to prevent geometry distortion; rely on LJ repulsion and steric clash penalty instead.
            # Van der Waals radii for overlap resolution (Angstroms)
            # vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52,
            #              'F': 1.47, 'Cl': 1.75, 'Br': 1.83, 'I': 1.98,
            #              'S': 1.80, 'At': 2.02}
            # for _iteration in range(10):  # Multiple passes to resolve cascading overlaps
            #     any_overlap = False
            #     for la in ligand_atoms:
            #         # Skip dummy atoms (they represent virtual charge sites, not steric atoms)
            #         if la.get('is_dummy', False):
            #             continue
            #     for ra in receptor_atoms:
            #         if ra.get('is_dummy', False):
            #             continue
            #         ddx = la['x'] - ra['x']
            #         ddy = la['y'] - ra['y']
            #         ddz = la['z'] - ra['z']
            #         dd = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
            #         if dd > 0.01:  # Avoid division by zero
            #             # Calculate sum of vdW radii for this atom pair
            #             radius_sum = vdw_radii.get(la['element'], 1.7) + vdw_radii.get(ra['element'], 1.7)
            #             # Use 80% of vdW sum as minimum distance (allows slight overlap)
            #             min_distance = radius_sum * 0.8
            #             if dd < min_distance:
            #                 # Push ligand atom outward along the overlap vector
            #                 shift = (min_distance - dd) / dd * 0.8  # 80% correction per pass
            #                 la['x'] += ddx * shift
            #                 la['y'] += ddy * shift
            #                 la['z'] += ddz * shift
            #                 any_overlap = True
            #     if not any_overlap:
            #         break
            total_energy = 0.0
            total_lj = 0.0
            total_coulomb = 0.0
            pairs_count = 0
            steric_clash_detected = False

            # Pre-find halogen and bonded carbon for directional Coulomb correction
            halogen_atom = None
            bonded_carbon = None
            for _a in ligand_atoms:
                if _a['element'] in ['F', 'Cl', 'Br', 'I', 'At']:
                    halogen_atom = _a
                    break
            if halogen_atom is not None:
                bonded_carbon = self._find_bonded_carbon(halogen_atom, ligand_atoms)

            # Diagnostic tracking for key distances
            min_distance = float('inf')
            min_halogen_o_distance = float('inf')
            min_dummy_o_distance = float('inf')
            has_halogen = False
            has_oxygen = False
            has_dummy = False

            # Calculate interactions between all ligand and receptor atoms
            for lig_atom in ligand_atoms:
                for rec_atom in receptor_atoms:
                    # Initialize energy variables to avoid UnboundLocalError
                    lj_energy = 0.0
                    coulomb_energy = 0.0
                    # Skip if both are dummy atoms or both are hydrogens (optional)
                    # We want to include all interactions for scoring

                    # Calculate distance
                    dx = lig_atom['x'] - rec_atom['x']
                    dy = lig_atom['y'] - rec_atom['y']
                    dz = lig_atom['z'] - rec_atom['z']
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)

                    # Track minimum distance for diagnostics
                    if distance < min_distance:
                        min_distance = distance

                    # Identify special atoms for sigma-hole diagnostics
                    lig_is_halogen = lig_atom['element'] in ['F', 'Cl', 'Br', 'I', 'At']
                    lig_is_dummy = lig_atom.get('is_dummy', False)
                    rec_is_oxygen = (rec_atom['element'] == 'O')

                    if lig_is_halogen:
                        has_halogen = True
                    if lig_is_dummy:
                        has_dummy = True
                    if rec_is_oxygen:
                        has_oxygen = True

                    # Track key distances for sigma-hole interactions
                    if lig_is_halogen and rec_is_oxygen and distance < min_halogen_o_distance:
                        min_halogen_o_distance = distance
                    if lig_is_dummy and rec_is_oxygen and distance < min_dummy_o_distance:
                        min_dummy_o_distance = distance

                    # Skip if too far apart (to avoid negligible interactions)
                    # Note: We don't set a lower cutoff here because:
                    # 1. Division by zero is handled by the distance > 0 checks below
                    # 2. The LJ potential correctly handles close-range repulsion
                    # 3. Setting a lower cutoff (like 2.8 Å) would exclude favorable vdW interactions
                    if distance > cutoff_distance:
                        continue

                    # Initialize energy variables to avoid UnboundLocalError
                    lj_energy = 0.0
                    coulomb_energy = 0.0

                    # Lennard-Jones (Van der Waals)
                    # Check if either atom is a dummy atom (virtual charge site)
                    # Dummy atoms have reduced LJ parameters for steric repulsion
                    is_lig_dummy = lig_atom.get('is_dummy', False)
                    is_rec_dummy = rec_atom.get('is_dummy', False)

                    if is_lig_dummy or is_rec_dummy:
                        # Dummy atoms need steric repulsion to prevent receptor atoms
                        # from overlapping them. Use small LJ with sigma=1.2 A.
                        epsilon, sigma = 0.02, 1.2
                    else:
                        epsilon, sigma = self._get_lj_parameters(
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
                        # Cap per-pair LJ repulsion to prevent energy explosion from overlapping atoms
                        if lj_energy > 10.0:
                            lj_energy = 10.0
                    else:
                        lj_energy = 0.0
                    total_lj += lj_energy

                    # Coulomb (Electrostatics) with directional corrections
                    if distance > 0:
                        epsilon_r = max(self.dielectric_coeff, 1.0)

                        # Determine charge scale factor for this pair
                        charge_factor = 1.0
                        _dbg_cf = charge_factor  # trace

                        if lig_atom is halogen_atom:
                            # Halogen-acceptor: suppress in sigma-hole direction
                            angle = self._compute_cx_acceptor_angle(halogen_atom, rec_atom, ligand_atoms)
                            charge_factor = self._halogen_acceptor_charge_scale(angle)
                        elif bonded_carbon is not None and lig_atom is bonded_carbon:
                            # Bonded carbon-acceptor: suppress in sigma-hole direction
                            angle = self._compute_cx_acceptor_angle(halogen_atom, rec_atom, ligand_atoms)
                            charge_factor = self._bonded_carbon_charge_scale(angle)
                        elif is_lig_dummy:
                            # Dummy atom: only interact with electronegative acceptors
                            charge_factor = self._dummy_acceptor_charge_scale(rec_atom["element"])

                        coulomb_energy = (self.k_coulomb * lig_atom["charge"] * rec_atom["charge"]
                            / (epsilon_r * distance)) * charge_factor
                        total_coulomb += coulomb_energy
                    else:
                        coulomb_energy = 0.0
                        total_coulomb += coulomb_energy

                # Total pairwise energy
                pair_energy = lj_energy + coulomb_energy
                total_energy += pair_energy
                pairs_count += 1
            logger.info(f"Physics score: {total_energy:.4f} kcal/mol from {pairs_count} atom pairs")
            logger.info(f"LJ contribution: {total_lj:.4f} kcal/mol, Coulomb contribution: {total_coulomb:.4f} kcal/mol")

            # Log diagnostic distance information
            if min_distance != float('inf'):
                logger.debug(f"Minimum ligand-receptor distance: {min_distance:.3f} Å")
            if has_halogen and has_oxygen and min_halogen_o_distance != float('inf'):
                logger.debug(f"Closest halogen-oxygen distance: {min_halogen_o_distance:.3f} Å")
            if has_dummy and has_oxygen and min_dummy_o_distance != float('inf'):
                logger.debug(f"Closest dummy-oxygen distance: {min_dummy_o_distance:.3f} Å")

            # Add sanity checks for unrealistic energies
            if total_energy > 0:
                logger.warning(f"Positive physics-based energy ({total_energy:.4f} kcal/mol) suggests repulsion rather than attraction")
            if abs(total_energy) > 50:
                logger.warning(f"Unphysically large energy magnitude ({total_energy:.4f} kcal/mol) - check for errors")

            # Cap Lennard-Jones repulsion to prevent energy explosions
            # Cap LJ contribution at 500 kcal/mol to prevent unrealistic energies from steric clashes
            if total_lj > 500.0:
                logger.warning(f"WARN: Steric clash detected, LJ contribution capped from {total_lj:.4f} to 500.00 kcal/mol")
                total_lj = 500.0
                total_energy = total_lj + total_coulomb  # Recalculate total energy with capped LJ

            # Additional steric clash detection: check for unphysically close contacts
            # Dummy atoms (H with positive charge) are included with reduced vdW radius
            vdw_radii = {'H': 1.2, 'HD': 0.6, 'C': 1.7, 'N': 1.55, 'O': 1.52,
                          'F': 1.57, 'Cl': 1.85, 'Br': 1.93, 'I': 2.08,
                          'S': 1.80, 'At': 2.02}
            clash_scale_factor = 0.6

            min_ratio = float('inf')
            min_pair = None
            min_dist = float('inf')
            min_threshold = float('inf')

            for lig_atom in ligand_atoms:
                lig_elem = 'HD' if lig_atom.get('is_dummy', False) else lig_atom['element']
                for rec_atom in receptor_atoms:
                    rec_elem = 'HD' if rec_atom.get('is_dummy', False) else rec_atom['element']
                    dx = lig_atom['x'] - rec_atom['x']
                    dy = lig_atom['y'] - rec_atom['y']
                    dz = lig_atom['z'] - rec_atom['z']
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if dist > 0.01:
                        radius_sum = vdw_radii.get(lig_elem, 1.7) + vdw_radii.get(rec_elem, 1.7)
                        threshold_pair = radius_sum * clash_scale_factor
                        ratio = dist / threshold_pair
                        # Skip H-H pairs - they're expected to be close during sigma-hole alignment
                        # and shouldn't contribute to steric penalties in our simplified model
                        is_hydrogen_pair = (lig_elem == 'H' and rec_elem == 'H')
                        if ratio < min_ratio and not is_hydrogen_pair:
                            min_ratio = ratio
                            min_pair = (lig_atom, rec_atom)
                            min_dist = dist
                            min_threshold = threshold_pair

            # Check if we found a clash (distance < threshold)
            # Only report clashes for non-hydrogen atom pairs
            if min_pair is not None and min_ratio < 1.0:
                lig_atom, rec_atom = min_pair
                logger.warning(f"WARN: Steric clash detected - {lig_atom['element']} and {rec_atom['element']} distance = {min_dist:.3f} Å < threshold {min_threshold:.3f} Å")
                # Proportional steric clash penalty: stronger when atoms overlap more
                # For sigma-hole models, penalty is small since the focus is on electrostatic attraction
                overlap = 1.0 - min_ratio
                steric_penalty = min(5.0 * overlap * overlap, 5.0)  # Capped at 5 kcal/mol
                logger.warning(f"Steric clash penalty: {steric_penalty:.2f} kcal/mol")
                total_energy += steric_penalty
                total_lj += steric_penalty
                steric_clash_detected = True
            # Check if energy calculation resulted in NaN (indicating failure upstream)
            if not np.isfinite(total_energy):
                logger.error(f"Physics-based scoring resulted in non-finite energy: {total_energy}")
                return (float('nan'), False)  # (energy, steric_clash_detected)

            # Ensure final binding energies stay in physically plausible range
            # For gas-phase sigma-hole interactions, typical range is -0.5 to -30 kcal/mol
            # Cap extreme values to prevent outliers
            if total_energy < -100.0:  # Unphysically strong binding
                logger.warning(f"Capping unphysically strong binding energy from {total_energy:.4f} to -100.00 kcal/mol")
                total_energy = -100.0
            elif total_energy > 50.0:  # Unphysically repulsive
                logger.warning(f"Capping unphysically repulsive energy from {total_energy:.4f} to 50.00 kcal/mol")
                total_energy = 50.0

            import sys; print(f'[FINAL] total={total_energy:.4f} lj={total_lj:.4f} coul={total_coulomb:.4f} pairs={pairs_count}', file=sys.stderr)
            return (total_energy, steric_clash_detected)  # (energy, steric_clash_detected)
        except Exception as e:
            import traceback; traceback.print_exc()
            logger.error(f"Error in physics-based scoring: {e}")
            return (float('nan'), False)

    def _parse_pdbqt(self, pdbqt_path: str) -> List[Dict]:
        """
        Parse PDBQT file to extract atom information.
        Handles both ATOM and HETATM records (for OpenBabel compatibility).

        Returns:
            List of dictionaries with keys: 'element', 'x', 'y', 'z', 'charge'
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

                                # residue_name = parts[3]
                                # chain_id = parts[4]
                                # residue_seq = int(parts[5])
                                x = float(parts[6])
                                y = float(parts[7])
                                z = float(parts[8])
                                # occupancy = float(parts[9])
                                # temperature_factor = float(parts[10])  # Usually 0.00 in PDBQT
                                charge = float(parts[11])  # Charge is at index 11 in PDBQT format
                                # Determine if this is a dummy atom (virtual charge site)
                                # Dummy atoms often have atom_type starting with 'HD' or have element H with positive charge
                                is_dummy = (atom_type == "EP")

                                # Apply charge_scale to dummy atom charges
                                if is_dummy and self.charge_scale != 1.0:
                                    charge *= self.charge_scale

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

    def run_vina_docking(self, receptor_pdbqt: str, ligand_pdbqt: str,
                        scoring: str = 'vinardo', exhaustiveness: int = 8,
                        num_modes: int = 9,
                        center_x: Optional[float] = None, center_y: Optional[float] = None, center_z: Optional[float] = None,
                        size_x: Optional[float] = None, size_y: Optional[float] = None, size_z: Optional[float] = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Run AutoDock Vina docking.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_pdbqt: Path to ligand PDBQT file
            scoring: Scoring function ('vinardo', 'ad4', 'vina')
            exhaustiveness: Exhaustiveness of search
            num_modes: Number of binding modes to generate
            center_x, center_y, center_z: Grid box center coordinates (Å)
            size_x, size_y, size_z: Grid box size dimensions (Å)

        Returns:
            Tuple of (best_affinity, output_text) or (None, error_message)
        """
        try:
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pdbqt = os.path.join(temp_dir, "output.pdbqt")
                log_file = os.path.join(temp_dir, "vina.log")

                # Build Vina command
                cmd = [
                    'vina',
                    '--receptor', receptor_pdbqt,
                    '--ligand', ligand_pdbqt,
                    '--out', output_pdbqt,
                    '--log', log_file,
                    '--scoring', scoring,
                    '--exhaustiveness', str(exhaustiveness),
                    '--num_modes', str(num_modes)
                ]

                # Add grid box parameters if provided
                if center_x is not None and center_y is not None and center_z is not None:
                    cmd.extend(['--center_x', str(center_x)])
                    cmd.extend(['--center_y', str(center_y)])
                    cmd.extend(['--center_z', str(center_z)])
                if size_x is not None and size_y is not None and size_z is not None:
                    cmd.extend(['--size_x', str(size_x)])
                    cmd.extend(['--size_y', str(size_y)])
                    cmd.extend(['--size_z', str(size_z)])

                logger.debug(f"Running Vina command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.error(f"Vina failed: {result.stderr}")
                    return None, f"Vina error: {result.stderr}"

                # Parse output log for affinity
                affinity = self._parse_vina_affinity(log_file)

                if affinity is not None:
                    logger.info(f"Vina docking completed. Best affinity: {affinity:.4f} kcal/mol")
                    return affinity, result.stdout
                else:
                    logger.warning("Could not parse affinity from Vina output")
                    return None, result.stdout

        except subprocess.TimeoutExpired:
            logger.error("Vina docking timed out after 120 seconds")
            return None, "Vina timeout"
        except FileNotFoundError:
            logger.error("Vina not found. Please install Vina or use Smina. Check that 'vina' is in your PATH.")
            return None, "Vina executable not found"
        except Exception as e:
            logger.error(f"Error running Vina docking: {e}", exc_info=True)
            return None, f"Docking error: {str(e)}"

    def run_smina_docking(self, receptor_pdbqt: str, ligand_pdbqt: str,
                         scoring: str = 'vinardo', exhaustiveness: int = 8,
                         num_modes: int = 9,
                         center_x: Optional[float] = None, center_y: Optional[float] = None, center_z: Optional[float] = None,
                         size_x: Optional[float] = None, size_y: Optional[float] = None, size_z: Optional[float] = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Run Smina docking (often better than Vina for custom scoring).

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_pdbqt: Path to ligand PDBQT file
            scoring: Scoring function
            exhaustiveness: Exhaustiveness of search
            num_modes: Number of binding modes to generate
            center_x, center_y, center_z: Grid box center coordinates (Å)
            size_x, size_y, size_z: Grid box size dimensions (Å)

        Returns:
            Tuple of (best_affinity, output_text) or (None, error_message)
        """
        try:
            # Check if smina is available
            subprocess.run(['smina', '--help'], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Smina not found, falling back to Vina")
            return self.run_vina_docking(receptor_pdbqt, ligand_pdbqt, scoring, exhaustiveness, num_modes)

        try:
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pdbqt = os.path.join(temp_dir, "output.pdbqt")
                log_file = os.path.join(temp_dir, "smina.log")

                # Build Smina command
                cmd = [
                    'smina',
                    '--receptor', receptor_pdbqt,
                    '--ligand', ligand_pdbqt,
                    '--out', output_pdbqt,
                    '--log', log_file,
                    '--scoring', scoring,
                    '--exhaustiveness', str(exhaustiveness),
                    '--num_modes', str(num_modes)
                ]

                # Add grid box parameters if provided
                if center_x is not None and center_y is not None and center_z is not None:
                    cmd.extend(['--center_x', str(center_x)])
                    cmd.extend(['--center_y', str(center_y)])
                    cmd.extend(['--center_z', str(center_z)])
                if size_x is not None and size_y is not None and size_z is not None:
                    cmd.extend(['--size_x', str(size_x)])
                    cmd.extend(['--size_y', str(size_y)])
                    cmd.extend(['--size_z', str(size_z)])

                logger.debug(f"Running Smina command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.error(f"Smina failed: {result.stderr}")
                    return None, f"Smina error: {result.stderr}"

                # Parse output log for affinity
                affinity = self._parse_vina_affinity(log_file)  # Same parser works for Smina

                if affinity is not None:
                    logger.info(f"Smina docking completed. Best affinity: {affinity:.4f} kcal/mol")
                    return affinity, result.stdout
                else:
                    logger.warning("Could not parse affinity from Smina output")
                    return None, result.stdout

        except subprocess.TimeoutExpired:
            logger.error("Smina docking timed out after 120 seconds")
            return None, "Smina timeout"
        except Exception as e:
            logger.error(f"Error running Smina docking: {e}", exc_info=True)
            return None, f"Docking error: {str(e)}"

    def _parse_vina_affinity(self, log_file: str) -> Optional[float]:
        """
        Parse Vina/Smina log file to extract binding affinity.

        Args:
            log_file: Path to Vina/Smina log file

        Returns:
            Best affinity (kcal/mol) or None if not found
        """
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Writing output' in line or 'Refined' in line:
                        # Look for affinity in docking output
                        continue
                    if 'Affinity:' in line:
                        # Format: "Affinity: -7.52 (kcal/mol)"
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                return float(parts[1])
                            except ValueError:
                                continue
                    elif 'REMARK   Vina Result:' in line:
                        # Alternative format in some versions
                        parts = line.split()
                        if len(parts) > 3:
                            try:
                                return float(parts[3])
                            except ValueError:
                                continue
        except Exception as e:
            logger.debug(f"Error parsing affinity from {log_file}: {e}")

        return None

    def compute_receptor_center(self, receptor_pdbqt: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
            logger.error(f"Failed to parse receptor PDBQT file for center computation: {receptor_pdbqt}")
            return None, None, None

        # Extract coordinates
        coords = []
        for atom in receptor_atoms:
            coords.append([atom['x'], atom['y'], atom['z']])

        if coords:
            coords_array = np.array(coords)
            center = coords_array.mean(axis=0)
            return float(center[0]), float(center[1]), float(center[2])
        else:
            # This should not happen if receptor_atoms is not empty, but just in case
            logger.error(f"No coordinates found in parsed receptor atoms: {receptor_pdbqt}")
            return None, None, None

    def score_only(self, receptor_pdbqt: str, ligand_pdbqt: str,
                  method: str = 'auto') -> float:
        """
        Calculate binding energy using score_only mode.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_pdbqt: Path to ligand PDBQT file
            method: Scoring method ('auto', 'vinardo', 'ad4', 'vina', 'smina', 'physics')

        Returns:
            Binding energy in kcal/mol
        """
        # Try physics-based first if requested or as reliable fallback
        if method == 'physics' or (method == 'auto' and self.use_physics_fallback):
            logger.info("Using physics-based scoring")
            result = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            # Extract energy from tuple (new format) or use directly (backward compatibility)
            if isinstance(result, tuple):
                return result[0]  # Return energy component
            else:
                return result  # Backward compatibility: assume it's just the energy

        # Try Vina/Smina scoring
        if method in ['auto', 'vinardo', 'ad4', 'vina']:
            scoring = method if method != 'auto' else 'vinardo'  # Default to vinardo for auto
            affinity, _ = self.run_vina_docking(
                receptor_pdbqt, ligand_pdbqt,
                scoring=scoring,
                exhaustiveness=8,
                num_modes=1  # Just need one mode for scoring
            )
            if affinity is not None:
                return affinity
            logger.warning(f"Vina {scoring} scoring failed or returned None")

        # Try Smina
        if method in ['auto', 'smina']:
            affinity, _ = self.run_smina_docking(
                receptor_pdbqt, ligand_pdbqt,
                scoring='vinardo',
                exhaustiveness=8,
                num_modes=1
            )
            if affinity is not None:
                return affinity
            logger.warning("Smina scoring failed or returned None")

        # Final fallback to physics-based
        if self.use_physics_fallback:
            logger.info("Falling back to physics-based scoring")
            result = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            # Extract energy from tuple (new format) or use directly (backward compatibility)
            if isinstance(result, tuple):
                return result[0]
            else:
                return result  # Backward compatibility

        logger.error("All scoring methods failed")
        return 0.0

    def dock_and_score(self, receptor_pdbqt: str, ligand_pdbqt: str,
                      scoring: str = 'vinardo',
                      exhaustiveness: int = 8,
                      num_modes: int = 9,
                      center_x: Optional[float] = None, center_y: Optional[float] = None, center_z: Optional[float] = None,
                      size_x: Optional[float] = None, size_y: Optional[float] = None, size_z: Optional[float] = None) -> Dict:
        """
        Perform docking and return comprehensive results.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_pdbqt: Path to ligand PDBQT file
            scoring: Scoring function to use
            exhaustiveness: Exhaustiveness of search
            num_modes: Number of binding modes to generate
            center_x, center_y, center_z: Grid box center coordinates (Å)
            size_x, size_y, size_z: Grid box size dimensions (Å)

        Returns:
            Dictionary with docking results
        """
        results = {
            'success': False,
            'best_affinity': None,
            'all_affinities': [],
            'binding_modes': [],
            'error': None
        }

        # Handle physics-based scoring directly
        if scoring == 'physics':
            logger.info("Using physics-based scoring")
            result = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            # Extract energy from tuple (new format) or use directly (backward compatibility)
            if isinstance(result, tuple):
                physics_energy = result[0]
            else:
                physics_energy = result  # Backward compatibility
            results['success'] = True
            results['best_affinity'] = physics_energy
            results['all_affinities'] = [physics_energy]
            results['method'] = 'physics'
            return results

        try:
            # Try Vina first
            affinity, output = self.run_vina_docking(
                receptor_pdbqt, ligand_pdbqt,
                scoring=scoring,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                center_x=center_x, center_y=center_y, center_z=center_z,
                size_x=size_x, size_y=size_y, size_z=size_z
            )

            if affinity is not None:
                results['success'] = True
                results['best_affinity'] = affinity
                results['raw_output'] = output

                # Parse all affinities from output if available
                affinities = self._parse_all_affinities(output)
                if affinities:
                    results['all_affinities'] = affinities
                else:
                    results['all_affinities'] = [affinity] if affinity is not None else []

                return results

            # If Vina failed, try Smina
            logger.warning("Vina docking failed, trying Smina...")
            affinity, output = self.run_smina_docking(
                receptor_pdbqt, ligand_pdbqt,
                scoring=scoring,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                center_x=center_x, center_y=center_y, center_z=center_z,
                size_x=size_x, size_y=size_y, size_z=size_z
            )

            if affinity is not None:
                results['success'] = True
                results['best_affinity'] = affinity
                results['raw_output'] = output

                affinities = self._parse_all_affinities(output)
                if affinities:
                    results['all_affinities'] = affinities
                else:
                    results['all_affinities'] = [affinity] if affinity is not None else []

                return results

            # If both fail, use physics-based as last resort
            if self.use_physics_fallback:
                logger.warning("Both Vina and Smina failed, using physics-based scoring")
                result = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
                # Extract energy from tuple (new format) or use directly (backward compatibility)
                if isinstance(result, tuple):
                    physics_energy = result[0]
                else:
                    physics_energy = result  # Backward compatibility

                results['success'] = True
                results['best_affinity'] = physics_energy
                results['all_affinities'] = [physics_energy]
                results['method'] = 'physics_fallback'
                return results

            # Everything failed
            results['error'] = "All docking/scoring methods failed"
            return results

        except Exception as e:
            logger.error(f"Error in dock_and_score: {e}")
            results['error'] = str(e)
            return results

    def _parse_all_affinities(self, vina_output: str) -> List[float]:
        """
        Parse all binding affinities from Vina/Smina output.

        Args:
            vina_output: Output text from Vina/Smina

        Returns:
            List of affinities (kcal/mol)
        """
        affinities = []
        try:
            lines = vina_output.split('\n')
            for line in lines:
                if 'mode' in line.lower() and 'affinity' in line.lower():
                    # Look for lines like: "  1      -7.52      -7.43      -7.41"
                    parts = line.split()
                    # Look for numeric values that could be affinities
                    for part in parts:
                        try:
                            val = float(part)
                            # Affinities are typically negative and in reasonable range
                            if -20.0 <= val <= 0.0:
                                affinities.append(val)
                        except ValueError:
                            continue
        except Exception as e:
            logger.debug(f"Error parsing affinities: {e}")

        return affinities

    def batch_score(self, receptor_pdbqt: str, ligand_dir: str,
                   output_csv: str,
                   method: str = 'auto',
                   exhaustiveness: int = 8,
                   num_modes: int = 9,
                   center_x: Optional[float] = None, center_y: Optional[float] = None, center_z: Optional[float] = None,
                   size_x: Optional[float] = None, size_y: Optional[float] = None, size_z: Optional[float] = None) -> pd.DataFrame:
        """
        Score multiple ligands against a single receptor with docking.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_dir: Directory containing ligand PDBQT files
            output_csv: Path to save results CSV
            method: Scoring method to use
            exhaustiveness: Exhaustiveness of search
            num_modes: Number of binding modes to generate
            center_x, center_y, center_z: Grid box center coordinates (Å)
            size_x, size_y, size_z: Grid box size dimensions (Å)

        Returns:
            DataFrame with scoring results
        """
        results = []

        # Get all PDBQT files in ligand directory
        ligand_files = [f for f in os.listdir(ligand_dir)
                       if f.endswith('.pdbqt') and os.path.isfile(os.path.join(ligand_dir, f))]

        logger.info(f"Found {len(ligand_files)} ligand files to score")

        for ligand_file in ligand_files:
            ligand_path = os.path.join(ligand_dir, ligand_file)
            ligand_name = ligand_file.replace('_ligand.pdbqt', '').replace('.pdbqt', '')

            try:
                # Perform actual docking instead of score_only
                docking_results = self.dock_and_score(
                    receptor_pdbqt=receptor_pdbqt,
                    ligand_pdbqt=ligand_path,
                    scoring=method,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    center_x=center_x, center_y=center_y, center_z=center_z,
                    size_x=size_x, size_y=size_y, size_z=size_z
                )

                if docking_results['success']:
                    affinity = docking_results['best_affinity']
                    steric_clash = False  # No steric clash info from Vina/Smina yet
                    if steric_clash:
                        logger.warning(f"STERIC CLASH detected for {ligand_name} during docking")
                else:
                    logger.warning(f"Docking failed for {ligand_name}: {docking_results.get('error', 'Unknown error')}")
                    # Fallback to physics-based scoring if enabled
                    if self.use_physics_fallback:
                        result = self.calculate_physics_score(ligand_path, receptor_pdbqt)
                        # Extract energy and steric clash info from tuple (new format) or use directly (backward compatibility)
                        if isinstance(result, tuple):
                            affinity, steric_clash = result
                        else:
                            affinity = result  # Backward compatibility
                            steric_clash = False  # Assume no clash info in backward compatibility mode
                    else:
                        affinity = float('nan')  # Return NaN instead of 0.0 to avoid silent failures
                        steric_clash = False

                    # Log steric clash detection if applicable
                    if steric_clash:
                        logger.warning(f"STERIC CLASH detected for {ligand_name} during physics-based fallback scoring")

                # Check for problematic affinity values and log appropriately
                if not np.isfinite(affinity):
                    logger.warning(f"Scoring for {ligand_name} produced non-finite result: {affinity}")
                elif affinity == 0.0:
                    logger.warning(f"Scoring for {ligand_name} returned exactly 0.0 kcal/mol - this may indicate a problem")

                results.append({
                    'compound_id': ligand_name,
                    'ligand_file': ligand_file,
                    'binding_energy_kcalmol': affinity,
                    'scoring_method': method,
                    'steric_clash': steric_clash
                })
                if steric_clash:
                    logger.warning(f"Result for {ligand_name} marked as STERIC CLASH in final output")

                if np.isfinite(affinity):
                    logger.debug(f"Scored {ligand_name}: {affinity:.4f} kcal/mol")
                else:
                    logger.debug(f"Scored {ligand_name}: {affinity} (non-finite)")

            except Exception as e:
                logger.error(f"Error scoring {ligand_name}: {e}")
                # Instead of returning 0.0 silently, return NaN to make the error visible
                results.append({
                    'compound_id': ligand_name,
                    'ligand_file': ligand_file,
                    'binding_energy_kcalmol': float('nan'),
                    'scoring_method': method,
                    'error': str(e),
                    'steric_clash': False
                })

        # Create DataFrame and save
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        logger.info(f"Saved scoring results to {output_csv}")

        return df_results

def example_usage():
    """Example usage of the docking engine."""
    engine = SigmaHoleDockingEngine()

    print("Sigma Hole Docking Engine Example")
    print("=" * 40)

    # Example 1: Physics-based scoring (doesn't require external tools)
    print("\n1. Physics-based scoring example:")
    # We'll use the files created earlier if they exist
    ligand_file = "iodobenzene_sigma.pdbqt"
    receptor_file = "acetone_receptor.pdbqt"

    if os.path.exists(ligand_file) and os.path.exists(receptor_file):
        energy = engine.calculate_physics_score(ligand_file, receptor_file)
        print(f"Physics-based energy: {energy:.4f} kcal/mol")
    else:
        print("Example files not found. Run ligand_generator.py and receptor_processor.py first.")

    # Example 2: Score only method
    print("\n2. Score-only method:")
    if os.path.exists(ligand_file) and os.path.exists(receptor_file):
        energy = engine.score_only(receptor_file, ligand_file, method='auto')
        print(f"Score-only energy: {energy:.4f} kcal/mol")
    else:
        print("Skipping score-only - files not found")

    # Example 3: Full docking (will likely fail without Vina/Smina installed)
    print("\n3. Full docking attempt:")
    if os.path.exists(ligand_file) and os.path.exists(receptor_file):
        results = engine.dock_and_score(receptor_file, ligand_file, scoring='vinardo')
        if results['success']:
            print(f"Docking successful! Best affinity: {results['best_affinity']:.4f} kcal/mol")
            if 'method' in results:
                print(f"Method used: {results['method']}")
        else:
            print(f"Docking failed: {results['error']}")
    else:
        print("Skipping docking - files not found")

    return engine

if __name__ == "__main__":
    example_usage()