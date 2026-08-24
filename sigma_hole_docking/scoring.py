"""
Scoring functions for sigma-hole docking — LJ + Coulomb energy, charge scaling, and physics-based scoring.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScoringMixin:
    """Mixin: Lennard-Jones + Coulomb scoring and charge-scaling logic."""

    # Default Lennard-Jones parameters for common atom pairs.
    # Format: (epsilon, sigma) where epsilon is well depth (kcal/mol), sigma is distance (Å)
    lj_params = {
        ("O", "H"): (0.075, 2.0),
        ("O", "C"): (0.10, 3.5),
        ("O", "O"): (0.05, 3.0),
        ("C", "H"): (0.025, 2.5),
        ("C", "C"): (0.05, 3.4),
        ("O", "F"): (0.05, 2.8),
        ("O", "Cl"): (0.10, 3.1),
        ("O", "Br"): (0.15, 3.2),
        ("O", "I"): (0.20, 3.3),
        ("C", "F"): (0.05, 2.9),
        ("C", "Cl"): (0.10, 3.2),
        ("C", "Br"): (0.15, 3.3),
        ("C", "I"): (0.20, 3.5),
        ("N", "F"): (0.05, 2.8),
        ("N", "Cl"): (0.10, 3.1),
        ("N", "Br"): (0.15, 3.2),
        ("N", "I"): (0.20, 3.3),
        ("S", "F"): (0.05, 3.0),
        ("S", "Cl"): (0.10, 3.3),
        ("S", "Br"): (0.15, 3.4),
        ("S", "I"): (0.20, 3.6),
    }

    def _compute_cx_acceptor_angle(self, halogen_atom, acceptor_atom, ligand_atoms):
        """Compute C-X...Acceptor angle at the halogen vertex.
        Returns angle in degrees, or None if bonded carbon not found."""
        hal_pos = np.array([halogen_atom["x"], halogen_atom["y"], halogen_atom["z"]])
        hal_elem = halogen_atom["element"]
        cutoff = {"F": 1.8, "Cl": 2.2, "Br": 2.4, "I": 2.6, "At": 2.7}.get(hal_elem, 2.3) + 0.3
        min_c_dist = float("inf")
        carbon_atom = None
        for atom in ligand_atoms:
            if atom["element"] == "C":
                c_pos = np.array([atom["x"], atom["y"], atom["z"]])
                d = np.linalg.norm(hal_pos - c_pos)
                if d < min_c_dist and d < cutoff:
                    min_c_dist = d
                    carbon_atom = atom
        if carbon_atom is None:
            return None
        vec_xc = np.array(
            [
                carbon_atom["x"] - halogen_atom["x"],
                carbon_atom["y"] - halogen_atom["y"],
                carbon_atom["z"] - halogen_atom["z"],
            ]
        )
        vec_xa = np.array(
            [
                acceptor_atom["x"] - halogen_atom["x"],
                acceptor_atom["y"] - halogen_atom["y"],
                acceptor_atom["z"] - halogen_atom["z"],
            ]
        )
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
        if rec_element in ["O", "N", "S", "F"]:
            return 1.0
        else:
            return 0.0

    def _find_bonded_carbon(self, halogen_atom, ligand_atoms):
        """Find the carbon atom bonded to the halogen."""
        hal_pos = np.array([halogen_atom["x"], halogen_atom["y"], halogen_atom["z"]])
        hal_elem = halogen_atom["element"]
        cutoff = {"F": 1.8, "Cl": 2.2, "Br": 2.4, "I": 2.6, "At": 2.7}.get(hal_elem, 2.3) + 0.3
        min_c_dist = float("inf")
        carbon_atom = None
        for atom in ligand_atoms:
            if atom["element"] == "C":
                c_pos = np.array([atom["x"], atom["y"], atom["z"]])
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
            "H": (0.05, 1.5),
            "C": (0.10, 2.0),
            "N": (0.10, 1.8),
            "O": (0.15, 1.8),
            "S": (0.20, 2.0),
            "F": (0.15, 1.7),
            "Cl": (0.20, 2.0),
            "Br": (0.22, 2.1),
            "I": (0.25, 2.2),
        }

        eps1, sig1 = defaults.get(atom1, (0.10, 2.0))
        eps2, sig2 = defaults.get(atom2, (0.10, 2.0))

        epsilon = math.sqrt(eps1 * eps2)
        sigma = (sig1 + sig2) / 2

        return (epsilon, sigma)

    def _find_halogen_and_carbon(
        self, ligand_atoms: List[Dict]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Find halogen atom and the carbon bonded to it in ligand.

        Returns:
            Tuple of (halogen_atom, carbon_atom) where:
            - halogen_atom: Dict with halogen info (F, Cl, Br, I)
            - carbon_atom: Dict of carbon bonded to halogen, or None if not found
        """
        halogen_atom = None
        carbon_atoms = [atom for atom in ligand_atoms if atom["element"] == "C"]

        # Find halogen
        for atom in ligand_atoms:
            if atom["element"] in ["F", "Cl", "Br", "I", "At"]:
                halogen_atom = atom
                break

        if not halogen_atom:
            return None, None

        # Find closest carbon to halogen (assumed to be bonded)
        if carbon_atoms:
            min_dist = float("inf")
            closest_carbon = None
            for carbon in carbon_atoms:
                dist = np.sqrt(
                    (halogen_atom["x"] - carbon["x"]) ** 2
                    + (halogen_atom["y"] - carbon["y"]) ** 2
                    + (halogen_atom["z"] - carbon["z"]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_carbon = carbon
            return halogen_atom, closest_carbon

        return halogen_atom, None

    def _find_all_halogens_and_carbons(
        self, ligand_atoms: List[Dict]
    ) -> List[Tuple[Dict, Optional[Dict]]]:
        """
        Find ALL halogen atoms and their bonded carbons in the ligand.

        Returns:
            List of (halogen_atom, carbon_atom) tuples for every halogen (F, Cl, Br, I, At).
            carbon_atom may be None if no bonded carbon is found.
        """
        carbon_atoms = [atom for atom in ligand_atoms if atom["element"] == "C"]
        pairs = []

        for atom in ligand_atoms:
            if atom["element"] in ["F", "Cl", "Br", "I", "At"]:
                closest_carbon = None
                if carbon_atoms:
                    min_dist = float("inf")
                    for carbon in carbon_atoms:
                        dist = np.sqrt(
                            (atom["x"] - carbon["x"]) ** 2
                            + (atom["y"] - carbon["y"]) ** 2
                            + (atom["z"] - carbon["z"]) ** 2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_carbon = carbon
                pairs.append((atom, closest_carbon))

        return pairs

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
        acceptor_elements = ["O", "N", "S", "F"]

        acceptor_atoms = []
        for element in acceptor_elements:
            element_atoms = [atom for atom in receptor_atoms if atom["element"] == element]
            if element_atoms:
                acceptor_atoms.extend(element_atoms)
                # Log which acceptor type was found
                logger.info(f"Found {len(element_atoms)} {element} acceptor atom(s)")
                # Continue to collect all acceptor types (O, N, S, F)

        if not acceptor_atoms:
            logger.warning("No acceptor atoms (O/N/S/F) found in receptor")
            # Fallback to oxygen-only for backward compatibility
            acceptor_atoms = [atom for atom in receptor_atoms if atom["element"] == "O"]

        return acceptor_atoms

    def _calculate_pairwise_energy(
        self, ligand_atoms: List[Dict], receptor_atoms: List[Dict]
    ) -> float:
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
            if _a["element"] in ["F", "Cl", "Br", "I", "At"]:
                _halogen = _a
                break
        if _halogen is not None:
            _bonded_c = self._find_bonded_carbon(_halogen, ligand_atoms)

        # Calculate interactions between all ligand and receptor atoms
        for lig_atom in ligand_atoms:
            for rec_atom in receptor_atoms:
                # Calculate distance
                dx = lig_atom["x"] - rec_atom["x"]
                dy = lig_atom["y"] - rec_atom["y"]
                dz = lig_atom["z"] - rec_atom["z"]
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                # Skip if too far apart (to avoid negligible interactions)
                if distance > 6.0:
                    continue

                # Initialize energy variables to avoid UnboundLocalError
                lj_energy = 0.0
                coulomb_energy = 0.0

                # Lennard-Jones (Van der Waals)
                is_lig_dummy = lig_atom.get("is_dummy", False)
                is_rec_dummy = rec_atom.get("is_dummy", False)

                if is_lig_dummy or is_rec_dummy:
                    epsilon, sigma = 0.02, 1.2
                else:
                    epsilon, sigma = self._get_lj_parameters(
                        lig_atom["element"], rec_atom["element"]
                    )

                # Prevent division by zero or excessively small distances
                min_dist_clamp = max(0.6 * sigma, 0.5)
                if distance < min_dist_clamp:
                    distance = min_dist_clamp

                if distance > 0:
                    lj_ratio = sigma / distance
                    lj_term = lj_ratio**6
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
                        charge_factor = self._dummy_acceptor_charge_scale(rec_atom["element"])

                    coulomb_energy = (
                        self.k_coulomb
                        * lig_atom["charge"]
                        * rec_atom["charge"]
                        / (epsilon_r * distance)
                    ) * charge_factor
                else:
                    coulomb_energy = 0.0

                # Total pairwise energy
                pair_energy = lj_energy + coulomb_energy
                total_energy += pair_energy
                pairs_count += 1

    def calculate_physics_score(
        self, ligand_pdbqt: str, receptor_pdbqt: str, cutoff_distance: float = 6.0
    ) -> Tuple[float, bool]:
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
                logger.error(
                    f"Failed to parse ligand PDBQT file: {ligand_pdbqt} - no valid ATOM/HETATM records found"
                )
                return (float("nan"), False)
            logger.info(f"Parsed ligand: {len(ligand_atoms)} atoms from {ligand_pdbqt}")

            receptor_atoms = self._parse_pdbqt(receptor_pdbqt)
            if not receptor_atoms:
                logger.error(
                    f"Failed to parse receptor PDBQT file: {receptor_pdbqt} - no valid ATOM/HETATM records found"
                )
                return (float("nan"), False)
            logger.info(f"Parsed receptor: {len(receptor_atoms)} atoms from {receptor_pdbqt}")

            # Validate that receptor has at least one electronegative atom (O/N/S/F) for sigma-hole interactions
            # This is important because without acceptor atoms, sigma-hole scoring doesn't make sense
            electronegative_elements = {"O", "N", "S", "F"}
            electronegative_count = sum(
                1 for atom in receptor_atoms if atom["element"] in electronegative_elements
            )

            if electronegative_count == 0:
                logger.warning(
                    f"No electronegative atoms (O/N/S/F) found in receptor PDBQT: {receptor_pdbqt}"
                )
                logger.warning(
                    "Sigma-hole interactions require acceptor atoms (O/N/S/F). Returning NaN."
                )
                return (float("nan"), False)

            # Align ligand for sigma-hole interaction if halogen and oxygen are present
            logger.info(
                f"Aligning ligand: {len(ligand_atoms)} ligand atoms, {len(receptor_atoms)} receptor atoms"
            )
            ligand_atoms = self._align_molecules_for_sigma_hole(ligand_atoms, receptor_atoms)
            # Validate coordinates after alignment
            if not self._validate_coordinates(ligand_atoms, "after alignment"):
                logger.error("ALIGNMENT FAILED: Invalid coordinates detected after alignment")
                # Return NaN to indicate failure instead of proceeding with invalid coordinates
                return (float("nan"), False)

            # Perform local optimization to refine the pose and find energy minimum
            logger.info(
                f"Local optimizing pose: {len(ligand_atoms)} ligand atoms, {len(receptor_atoms)} receptor atoms"
            )
            ligand_atoms = self._local_optimize_pose(ligand_atoms, receptor_atoms)
            # Validate coordinates after optimization
            if not self._validate_coordinates(ligand_atoms, "after local optimization"):
                logger.error(
                    "OPTIMIZATION FAILED: Invalid coordinates detected after local optimization"
                )
                # Return NaN to indicate failure instead of proceeding with invalid coordinates
                return (float("nan"), False)

            # Compute geometric centers (for diagnostics only, no separation applied)
            def compute_center(atoms):
                if not atoms:
                    return 0.0, 0.0, 0.0
                sum_x = sum(atom["x"] for atom in atoms)
                sum_y = sum(atom["y"] for atom in atoms)
                sum_z = sum(atom["z"] for atom in atoms)
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

            # Pre-find ALL halogens and their bonded carbons for directional Coulomb correction
            halogen_carbon_pairs = self._find_all_halogens_and_carbons(ligand_atoms)
            all_halogen_ids = {id(h) for h, _ in halogen_carbon_pairs}
            carbon_to_halogen = {id(c): h for h, c in halogen_carbon_pairs if c is not None}
            # Keep first halogen for backward-compatible diagnostics
            halogen_carbon_pairs[0][0] if halogen_carbon_pairs else None
            halogen_carbon_pairs[0][1] if halogen_carbon_pairs else None

            # Diagnostic tracking for key distances
            min_distance = float("inf")
            min_halogen_o_distance = float("inf")
            min_dummy_o_distance = float("inf")
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
                    dx = lig_atom["x"] - rec_atom["x"]
                    dy = lig_atom["y"] - rec_atom["y"]
                    dz = lig_atom["z"] - rec_atom["z"]
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                    # Track minimum distance for diagnostics
                    if distance < min_distance:
                        min_distance = distance

                    # Identify special atoms for sigma-hole diagnostics
                    lig_is_halogen = lig_atom["element"] in ["F", "Cl", "Br", "I", "At"]
                    lig_is_dummy = lig_atom.get("is_dummy", False)
                    rec_is_oxygen = rec_atom["element"] == "O"

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
                    is_lig_dummy = lig_atom.get("is_dummy", False)
                    is_rec_dummy = rec_atom.get("is_dummy", False)

                    if is_lig_dummy or is_rec_dummy:
                        # Dummy atoms need steric repulsion to prevent receptor atoms
                        # from overlapping them. Use small LJ with sigma=1.2 A.
                        epsilon, sigma = 0.02, 1.2
                    else:
                        epsilon, sigma = self._get_lj_parameters(
                            lig_atom["element"], rec_atom["element"]
                        )

                    # Prevent division by zero or excessively small distances
                    min_dist_clamp = max(0.6 * sigma, 0.5)
                    if distance < min_dist_clamp:
                        distance = min_dist_clamp

                    if distance > 0:
                        lj_ratio = sigma / distance
                        lj_term = lj_ratio**6
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

                        if id(lig_atom) in all_halogen_ids:
                            # This ligand atom is a halogen — suppress its charge
                            # in the sigma-hole direction (angle > 140°)
                            angle = self._compute_cx_acceptor_angle(
                                lig_atom, rec_atom, ligand_atoms
                            )
                            charge_factor = self._halogen_acceptor_charge_scale(angle)
                        elif id(lig_atom) in carbon_to_halogen:
                            # This ligand atom is a carbon bonded to a halogen —
                            # suppress in sigma-hole direction to avoid double-counting
                            this_halogen = carbon_to_halogen[id(lig_atom)]
                            angle = self._compute_cx_acceptor_angle(
                                this_halogen, rec_atom, ligand_atoms
                            )
                            charge_factor = self._bonded_carbon_charge_scale(angle)
                        elif is_lig_dummy:
                            # Dummy atom: only interact with electronegative acceptors
                            charge_factor = self._dummy_acceptor_charge_scale(rec_atom["element"])

                        coulomb_energy = (
                            self.k_coulomb
                            * lig_atom["charge"]
                            * rec_atom["charge"]
                            / (epsilon_r * distance)
                        ) * charge_factor
                        total_coulomb += coulomb_energy
                    else:
                        coulomb_energy = 0.0
                        total_coulomb += coulomb_energy

            # Total energy is the sum of all LJ and Coulomb contributions
            # (total_lj and total_coulomb are accumulated inside the inner loop)
            total_energy = total_lj + total_coulomb
            pairs_count = len(ligand_atoms) * len(receptor_atoms)
            logger.info(f"Physics score: {total_energy:.4f} kcal/mol from {pairs_count} atom pairs")
            logger.info(
                f"LJ contribution: {total_lj:.4f} kcal/mol, Coulomb contribution: {total_coulomb:.4f} kcal/mol"
            )

            # Log diagnostic distance information
            if min_distance != float("inf"):
                logger.debug(f"Minimum ligand-receptor distance: {min_distance:.3f} Å")
            if has_halogen and has_oxygen and min_halogen_o_distance != float("inf"):
                logger.debug(f"Closest halogen-oxygen distance: {min_halogen_o_distance:.3f} Å")
            if has_dummy and has_oxygen and min_dummy_o_distance != float("inf"):
                logger.debug(f"Closest dummy-oxygen distance: {min_dummy_o_distance:.3f} Å")

            # Add sanity checks for unrealistic energies
            if total_energy > 0:
                logger.warning(
                    f"Positive physics-based energy ({total_energy:.4f} kcal/mol) suggests repulsion rather than attraction"
                )
            if abs(total_energy) > 50:
                logger.warning(
                    f"Unphysically large energy magnitude ({total_energy:.4f} kcal/mol) - check for errors"
                )

            # Cap Lennard-Jones repulsion to prevent energy explosions
            # Cap LJ contribution at 500 kcal/mol to prevent unrealistic energies from steric clashes
            if total_lj > 500.0:
                logger.warning(
                    f"WARN: Steric clash detected, LJ contribution capped from {total_lj:.4f} to 500.00 kcal/mol"
                )
                total_lj = 500.0
                total_energy = total_lj + total_coulomb  # Recalculate total energy with capped LJ

            # Additional steric clash detection: check for unphysically close contacts
            # Dummy atoms (H with positive charge) are included with reduced vdW radius
            vdw_radii = {
                "H": 1.2,
                "HD": 0.6,
                "C": 1.7,
                "N": 1.55,
                "O": 1.52,
                "F": 1.57,
                "Cl": 1.85,
                "Br": 1.93,
                "I": 2.08,
                "S": 1.80,
                "At": 2.02,
            }
            clash_scale_factor = 0.6

            min_ratio = float("inf")
            min_pair = None
            min_dist = float("inf")
            min_threshold = float("inf")

            for lig_atom in ligand_atoms:
                lig_elem = "HD" if lig_atom.get("is_dummy", False) else lig_atom["element"]
                for rec_atom in receptor_atoms:
                    rec_elem = "HD" if rec_atom.get("is_dummy", False) else rec_atom["element"]
                    dx = lig_atom["x"] - rec_atom["x"]
                    dy = lig_atom["y"] - rec_atom["y"]
                    dz = lig_atom["z"] - rec_atom["z"]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > 0.01:
                        radius_sum = vdw_radii.get(lig_elem, 1.7) + vdw_radii.get(rec_elem, 1.7)
                        threshold_pair = radius_sum * clash_scale_factor
                        ratio = dist / threshold_pair
                        # Skip H-H pairs - they're expected to be close during sigma-hole alignment
                        # and shouldn't contribute to steric penalties in our simplified model
                        is_hydrogen_pair = lig_elem == "H" and rec_elem == "H"
                        if ratio < min_ratio and not is_hydrogen_pair:
                            min_ratio = ratio
                            min_pair = (lig_atom, rec_atom)
                            min_dist = dist
                            min_threshold = threshold_pair

            # Check if we found a clash (distance < threshold)
            # Only report clashes for non-hydrogen atom pairs
            if min_pair is not None and min_ratio < 1.0:
                lig_atom, rec_atom = min_pair
                logger.warning(
                    f"WARN: Steric clash detected - {lig_atom['element']} and {rec_atom['element']} distance = {min_dist:.3f} Å < threshold {min_threshold:.3f} Å"
                )
                # Proportional steric clash penalty: stronger when atoms overlap more
                # For sigma-hole models, penalty is small since the focus is on electrostatic attraction
                overlap = 1.0 - min_ratio
                steric_penalty = min(5.0 * overlap * overlap, 5.0)  # Capped at 5 kcal/mol
                logger.warning(f"Steric clash penalty: {steric_penalty:.2f} kcal/mol")
                total_energy += steric_penalty
                total_lj += steric_penalty
            # Check if energy calculation resulted in NaN (indicating failure upstream)
            if not np.isfinite(total_energy):
                logger.error(f"Physics-based scoring resulted in non-finite energy: {total_energy}")
                return (float("nan"), False)  # (energy, steric_clash_detected)

            # Ensure final binding energies stay in physically plausible range
            # For gas-phase sigma-hole interactions, typical range is -0.5 to -30 kcal/mol
            # Cap extreme values to prevent outliers
            if total_energy < -100.0:  # Unphysically strong binding
                logger.warning(
                    f"Capping unphysically strong binding energy from {total_energy:.4f} to -100.00 kcal/mol"
                )
                total_energy = -100.0
            elif total_energy > 50.0:  # Unphysically repulsive
                logger.warning(
                    f"Capping unphysically repulsive energy from {total_energy:.4f} to 50.00 kcal/mol"
                )
                total_energy = 50.0

            return (total_energy, True)  # (energy, success)
        except Exception as e:
            logger.error(f"Error in physics-based scoring: {e}")
            return (float("nan"), False)

    def _parse_vina_affinity(self, log_file: str) -> Optional[float]:
        """
        Parse Vina/Smina log file to extract binding affinity.

        Args:
            log_file: Path to Vina/Smina log file

        Returns:
            Best affinity (kcal/mol) or None if not found
        """
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if "Writing output" in line or "Refined" in line:
                        # Look for affinity in docking output
                        continue
                    if "Affinity:" in line:
                        # Format: "Affinity: -7.52 (kcal/mol)"
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                return float(parts[1])
                            except ValueError:
                                continue
                    elif "REMARK   Vina Result:" in line:
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
            lines = vina_output.split("\n")
            for line in lines:
                if "mode" in line.lower() and "affinity" in line.lower():
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
