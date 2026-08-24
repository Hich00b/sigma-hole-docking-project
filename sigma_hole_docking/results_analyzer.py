"""
Sigma Hole Results Analyzer

Analyzes docking results, ranks compounds by sigma-hole interaction strength,
and generates validation reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os

from sigma_hole_docking.pdbqt_io import parse_pdbqt as _parse_pdbqt_file

logger = logging.getLogger(__name__)


class SigmaHoleResultsAnalyzer:
    """
    Analyzes and ranks sigma-hole docking results.
    """

    def __init__(self):
        """Initialize the results analyzer."""
        self.results_data = None

    def load_results(
        self, csv_path: str, energy_col: str = "binding_energy_kcalmol", id_col: str = "compound_id"
    ) -> pd.DataFrame:
        """
        Load docking results from CSV file.

        Args:
            csv_path: Path to results CSV file
            energy_col: Column name for binding energy values
            id_col: Column name for compound identifiers

        Returns:
            DataFrame with results data
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} results from {csv_path}")

            # Validate required columns
            if energy_col not in df.columns:
                raise ValueError(f"Energy column '{energy_col}' not found in CSV")
            if id_col not in df.columns:
                raise ValueError(f"ID column '{id_col}' not found in CSV")

            # Sort by binding energy (most negative = strongest binding)
            df_sorted = df.sort_values(by=energy_col, ascending=True).reset_index(drop=True)
            df_sorted["rank"] = range(1, len(df_sorted) + 1)

            self.results_data = df_sorted
            logger.info(f"Results sorted by {energy_col} (ascending)")
            return df_sorted

        except Exception as e:
            logger.error(f"Error loading results from {csv_path}: {e}")
            raise

    def rank_compounds(
        self,
        df: Optional[pd.DataFrame] = None,
        energy_col: str = "binding_energy_kcalmol",
        id_col: str = "compound_id",
    ) -> pd.DataFrame:
        """
        Rank compounds by binding energy (strongest first).

        Args:
            df: DataFrame with results (if None, uses self.results_data)
            energy_col: Column name for binding energy values
            id_col: Column name for compound identifiers

        Returns:
            DataFrame with ranked compounds
        """
        if df is None:
            if self.results_data is None:
                raise ValueError("No results data available. Load results first.")
            df = self.results_data.copy()
        else:
            df = df.copy()

        # Sort by binding energy (most negative = strongest binding)
        df_ranked = df.sort_values(by=energy_col, ascending=True).reset_index(drop=True)
        df_ranked["rank"] = range(1, len(df_ranked) + 1)

        # Add ranking categories
        df_ranked["binding_category"] = pd.cut(
            df_ranked[energy_col],
            bins=[-float("inf"), -5.0, -2.0, 0.0, float("inf")],
            labels=["Strong", "Moderate", "Weak", "Non-binding"],
            ordered=False,
        )

        logger.info(f"Ranked {len(df_ranked)} compounds")
        return df_ranked

    def get_top_hits(
        self,
        df: Optional[pd.DataFrame] = None,
        n: int = 10,
        energy_col: str = "binding_energy_kcalmol",
    ) -> pd.DataFrame:
        """
        Get top N compounds by binding energy.

        Args:
            df: DataFrame with results (if None, uses ranked results)
            n: Number of top compounds to return
            energy_col: Column name for binding energy values

        Returns:
            DataFrame with top N compounds
        """
        if df is None:
            ranked_df = self.rank_compounds()
        else:
            ranked_df = self.rank_compounds(df)

        top_hits = ranked_df.head(n).copy()
        logger.info(f"Retrieved top {min(n, len(top_hits))} hits")
        return top_hits

    def calculate_summary_statistics(
        self, df: Optional[pd.DataFrame] = None, energy_col: str = "binding_energy_kcalmol"
    ) -> Dict:
        """
        Calculate summary statistics for binding energies.

        Args:
            df: DataFrame with results (if None, uses self.results_data)
            energy_col: Column name for binding energy values

        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            if self.results_data is None:
                raise ValueError("No results data available. Load results first.")
            df = self.results_data
        else:
            df = df.copy()

        energies = df[energy_col].dropna()

        if len(energies) == 0:
            return {"count": 0}

        stats = {
            "count": len(energies),
            "mean": float(energies.mean()),
            "median": float(energies.median()),
            "std": float(energies.std()),
            "min": float(energies.min()),  # Most negative = strongest binding
            "max": float(energies.max()),  # Least negative/most positive = weakest binding
            "q25": float(energies.quantile(0.25)),
            "q75": float(energies.quantile(0.75)),
            "strong_binders": int((energies < -5.0).sum()),  # Arbitrary cutoff
            "moderate_binders": int(((energies >= -5.0) & (energies < -2.0)).sum()),
            "weak_binders": int(((energies >= -2.0) & (energies < 0.0)).sum()),
            "non_binders": int((energies >= 0.0).sum()),
        }

        logger.info(f"Calculated statistics for {stats['count']} compounds")
        return stats

    def validate_geometry(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt: str,
        halogen: str = "I",
        expected_distance_range: Tuple[float, float] = (2.8, 3.5),
        grid_box_center: Optional[Tuple[float, float, float]] = None,
        grid_box_size: Optional[Tuple[float, float, float]] = None,
        steric_clash_cutoff: float = 2.0,
    ) -> Dict:
        """
        Validate that dummy atom is positioned correctly for sigma-hole interaction.

        Args:
            ligand_pdbqt: Path to ligand PDBQT file
            receptor_pdbqt: Path to receptor PDBQT file
            halogen: Halogen element to validate
            expected_distance_range: Expected X...O distance range (Å) for sigma-hole
            grid_box_center: (x, y, z) coordinates of grid box center (Å)
            grid_box_size: (size_x, size_y, size_z) dimensions of grid box (Å)
            steric_clash_cutoff: Distance cutoff for steric clashes (Å)

        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": False,
            "halogen_found": False,
            "dummy_found": False,
            "distance": None,
            "distance_valid": False,
            "charge_valid": False,
            "halogen_position": None,
            "oxygen_position": None,
            "dummy_position": None,
            "error": None,
        }

        try:
            logger.info(
                f"Validating geometry for ligand {ligand_pdbqt} and receptor {receptor_pdbqt}"
            )
            logger.info(f"Looking for halogen: {halogen}")
            # Parse ligand to find halogen and dummy atoms
            ligand_atoms = self._parse_pdbqt_detailed(ligand_pdbqt)
            logger.info(f"Parsed {len(ligand_atoms)} atoms from ligand")
            halogen_atom = None
            dummy_atom = None

            for atom in ligand_atoms:
                if atom["element"] == halogen:
                    halogen_atom = atom
                # Dummy atom is typically hydrogen with positive charge
                elif atom["element"] == "H" and atom["charge"] > 0.01:
                    dummy_atom = atom

            # Parse receptor to find oxygen atoms (carbonyl oxygen typically)
            receptor_atoms = self._parse_pdbqt_detailed(receptor_pdbqt)
            oxygen_atoms = [atom for atom in receptor_atoms if atom["element"] == "O"]

            if not halogen_atom:
                validation["error"] = f"No {halogen} atom found in ligand"
                return validation

            validation["halogen_found"] = True
            validation["halogen_position"] = [
                halogen_atom["x"],
                halogen_atom["y"],
                halogen_atom["z"],
            ]

            if not dummy_atom:
                validation["error"] = "No dummy atom (H with positive charge) found in ligand"
                return validation

            validation["dummy_found"] = True
            validation["dummy_position"] = [dummy_atom["x"], dummy_atom["y"], dummy_atom["z"]]
            validation["dummy_charge"] = dummy_atom["charge"]
            validation["charge_valid"] = dummy_atom["charge"] > 0.01  # Reasonable positive charge

            if not oxygen_atoms:
                validation["error"] = "No oxygen atoms found in receptor"
                return validation

            # Find closest oxygen to halogen (for distance validation)
            min_distance = float("inf")
            closest_oxygen = None

            for ox_atom in receptor_atoms:
                if ox_atom["element"] == "O":
                    dx = halogen_atom["x"] - ox_atom["x"]
                    dy = halogen_atom["y"] - ox_atom["y"]
                    dz = halogen_atom["z"] - ox_atom["z"]
                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance < min_distance:
                        min_distance = distance
                        closest_oxygen = ox_atom

            if closest_oxygen is None:
                validation["error"] = "Could not determine closest oxygen"
                return validation

            validation["oxygen_position"] = [
                closest_oxygen["x"],
                closest_oxygen["y"],
                closest_oxygen["z"],
            ]
            validation["distance"] = min_distance

            # Check if distance is in expected range for sigma-hole
            low, high = expected_distance_range
            validation["distance_valid"] = low <= min_distance <= high

            # === ADDITIONAL VALIDATION CHECKS ===

            # 1. Grid box containment check
            validation["grid_box_containment"] = None
            validation["grid_box_valid"] = True  # Assume valid if no grid box defined
            if grid_box_center is not None and grid_box_size is not None:
                center_x, center_y, center_z = grid_box_center
                size_x, size_y, size_z = grid_box_size

                # Calculate grid box boundaries
                min_x = center_x - size_x / 2
                max_x = center_x + size_x / 2
                min_y = center_y - size_y / 2
                max_y = center_y + size_y / 2
                min_z = center_z - size_z / 2
                max_z = center_z + size_z / 2

                # Check if all ligand atoms are within grid box
                all_within = True
                for atom in ligand_atoms:
                    if not (
                        min_x <= atom["x"] <= max_x
                        and min_y <= atom["y"] <= max_y
                        and min_z <= atom["z"] <= max_z
                    ):
                        all_within = False
                        break

                validation["grid_box_containment"] = all_within
                validation["grid_box_valid"] = all_within

            # 2. Steric clash detection
            validation["steric_clashes"] = []
            validation["steric_clash_valid"] = True  # Assume valid if no clashes found
            if receptor_atoms and ligand_atoms:
                for lig_atom in ligand_atoms:
                    for rec_atom in receptor_atoms:
                        # Skip if same element and bonded (simplified: just check distance)
                        dx = lig_atom["x"] - rec_atom["x"]
                        dy = lig_atom["y"] - rec_atom["y"]
                        dz = lig_atom["z"] - rec_atom["z"]
                        distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                        # Consider it a steric clash if distance is less than cutoff
                        # and they're not covalently bonded (simplified check)
                        if distance < steric_clash_cutoff:
                            validation["steric_clashes"].append(
                                {
                                    "ligand_atom": lig_atom["element"],
                                    "receptor_atom": rec_atom["element"],
                                    "distance": distance,
                                    "ligand_position": [
                                        lig_atom["x"],
                                        lig_atom["y"],
                                        lig_atom["z"],
                                    ],
                                    "receptor_position": [
                                        rec_atom["x"],
                                        rec_atom["y"],
                                        rec_atom["z"],
                                    ],
                                }
                            )

                validation["steric_clash_valid"] = len(validation["steric_clashes"]) == 0

            # 3. Enhanced dummy atom orientation (check if pointing toward oxygen)
            validation["dummy_orientation_valid"] = True  # Default to valid
            if validation["dummy_found"] and validation["oxygen_position"] is not None:
                # Calculate vector from dummy to oxygen
                dummy_to_ox = np.array(
                    [
                        validation["oxygen_position"][0] - validation["dummy_position"][0],
                        validation["oxygen_position"][1] - validation["dummy_position"][1],
                        validation["oxygen_position"][2] - validation["dummy_position"][2],
                    ]
                )

                # For a proper sigma-hole, the dummy atom should be along the C-X bond extended
                # Since we don't have easy access to the bonded carbon, we'll check if
                # the dummy atom is in the general direction of the oxygen (positive dot product
                # with halogen->oxygen vector would indicate proper orientation)
                if validation["halogen_position"] is not None:
                    halogen_to_oxygen = np.array(
                        [
                            validation["oxygen_position"][0] - validation["halogen_position"][0],
                            validation["oxygen_position"][1] - validation["halogen_position"][1],
                            validation["oxygen_position"][2] - validation["halogen_position"][2],
                        ]
                    )

                    # Normalize vectors
                    dummy_to_ox_norm = np.linalg.norm(dummy_to_ox)
                    halogen_to_ox_norm = np.linalg.norm(halogen_to_oxygen)

                    if dummy_to_ox_norm > 0 and halogen_to_ox_norm > 0:
                        dummy_to_ox_unit = dummy_to_ox / dummy_to_ox_norm
                        halogen_to_ox_unit = halogen_to_oxygen / halogen_to_ox_norm

                        # Dot product should be positive for similar direction
                        # (dummy atom and oxygen should be roughly in same direction from halogen)
                        dot_product = np.dot(dummy_to_ox_unit, halogen_to_ox_unit)
                        validation["dummy_orientation_valid"] = (
                            dot_product > 0.5
                        )  # Reasonable threshold
                        validation["dummy_orientation_dot_product"] = float(dot_product)

            # 4. C-X···O angle validation (should be ≥ 160° for proper sigma-hole)
            validation["cx_o_angle"] = None
            validation["angle_valid"] = True  # Default to valid
            if (
                validation["halogen_found"]
                and validation["dummy_found"]
                and validation["oxygen_position"] is not None
            ):
                # Find carbon bonded to halogen by looking for carbon with reasonable distance to halogen
                ligand_atoms = self._parse_pdbqt_detailed(ligand_pdbqt)
                carbon_atom = None
                if validation["halogen_position"] is not None:
                    hx, hy, hz = validation["halogen_position"]
                    min_c_dist = float("inf")
                    for atom in ligand_atoms:
                        if atom["element"] == "C":
                            # Calculate distance to halogen
                            dist = np.sqrt(
                                (atom["x"] - hx) ** 2
                                + (atom["y"] - hy) ** 2
                                + (atom["z"] - hz) ** 2
                            )
                            # Typical C-X bond lengths: C-F ~1.3Å, C-Cl ~1.7Å, C-Br ~1.9Å, C-I ~2.1Å
                            if (
                                dist < min_c_dist and dist < 2.5
                            ):  # Reasonable upper bound for C-X bond
                                min_c_dist = dist
                                carbon_atom = atom

                if (
                    carbon_atom is not None
                    and validation["halogen_position"] is not None
                    and validation["oxygen_position"] is not None
                ):
                    # Calculate vectors for C-X···O angle
                    # Vector from halogen to carbon (C-X bond, pointing from halogen to carbon)
                    vec_xc = np.array(
                        [
                            carbon_atom["x"] - validation["halogen_position"][0],
                            carbon_atom["y"] - validation["halogen_position"][1],
                            carbon_atom["z"] - validation["halogen_position"][2],
                        ]
                    )
                    # Vector from halogen to oxygen (X···O interaction, pointing from halogen to oxygen)
                    vec_xo = np.array(
                        [
                            validation["oxygen_position"][0] - validation["halogen_position"][0],
                            validation["oxygen_position"][1] - validation["halogen_position"][1],
                            validation["oxygen_position"][2] - validation["halogen_position"][2],
                        ]
                    )

                    # Calculate angle between X-C and X-O vectors
                    norm_xc = np.linalg.norm(vec_xc)
                    norm_xo = np.linalg.norm(vec_xo)
                    if norm_xc > 0 and norm_xo > 0:
                        vec_xc_unit = vec_xc / norm_xc
                        vec_xo_unit = vec_xo / norm_xo
                        dot_product = np.dot(vec_xc_unit, vec_xo_unit)
                        # Clamp to [-1, 1] for numerical stability
                        dot_product = max(-1.0, min(1.0, dot_product))
                        angle_rad = np.arccos(dot_product)
                        angle_deg = np.degrees(angle_rad)
                        validation["cx_o_angle"] = float(angle_deg)
                        # For sigma-hole, we want C-X···O angle to be close to 180° (linear)
                        validation["angle_valid"] = (
                            angle_deg >= 160.0
                        )  # Allow some deviation from linear
                    else:
                        validation["angle_valid"] = False
                else:
                    validation["angle_valid"] = False  # Could not calculate angle

            # Overall validation including new checks
            validation["valid"] = (
                validation["halogen_found"]
                and validation["dummy_found"]
                and validation["charge_valid"]
                and validation["distance_valid"]
                and validation["grid_box_valid"]
                and validation["steric_clash_valid"]
                and validation["dummy_orientation_valid"]
                and validation["angle_valid"]
            )

            logger.info(f"Geometry validation: {'PASS' if validation['valid'] else 'FAIL'}")
            if validation["distance"] is not None:
                logger.info(
                    f"Halogen-Oxygen distance: {validation['distance']:.2f} Å "
                    f"(expected: {low}-{high} Å)"
                )
            if validation["grid_box_containment"] is not None:
                logger.info(
                    f"Grid box containment: {'PASS' if validation['grid_box_valid'] else 'FAIL'}"
                )
            if len(validation["steric_clashes"]) > 0:
                logger.info(f"Steric clashes found: {len(validation['steric_clashes'])}")
            if "dummy_orientation_dot_product" in validation:
                logger.info(
                    f"Dummy-Oxygen orientation dot product: {validation['dummy_orientation_dot_product']:.3f}"
                )
            if validation["cx_o_angle"] is not None:
                logger.info(f"C-X···O angle: {validation['cx_o_angle']:.1f}° (expected: ≥160.0°)")

        except Exception as e:
            logger.error(f"Error during geometry validation: {e}")
            validation["error"] = str(e)

        return validation

    def _parse_pdbqt_detailed(self, pdbqt_path: str) -> List[Dict]:
        """Parse PDBQT file using the consolidated :mod:`pdbqt_io` parser.

        Returns:
            List of dictionaries with atomic details
        """
        return _parse_pdbqt_file(pdbqt_path)

    def generate_ranking_report(
        self, df: Optional[pd.DataFrame] = None, output_dir: str = "ranking_reports"
    ) -> Dict[str, str]:
        """
        Generate comprehensive ranking report with plots and tables.

        Args:
            df: DataFrame with results (if None, uses ranked results)
            output_dir: Directory to save report files

        Returns:
            Dictionary mapping report type to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if df is None:
            df = self.rank_compounds()
        else:
            df = self.rank_compounds(df)

        report_files = {}

        try:
            # 1. Generate summary statistics
            stats = self.calculate_summary_statistics(df)
            stats_path = os.path.join(output_dir, f"summary_stats_{timestamp}.json")
            import json

            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            report_files["summary_stats"] = stats_path

            # 2. Generate top hits table
            top_hits = self.get_top_hits(df, n=20)
            top_hits_path = os.path.join(output_dir, f"top_hits_{timestamp}.csv")
            top_hits.to_csv(top_hits_path, index=False)
            report_files["top_hits"] = top_hits_path

            # 3. Generate energy distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(df["binding_energy_kcalmol"], bins=30, edgecolor="black", alpha=0.7)
            plt.xlabel("Binding Energy (kcal/mol)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Binding Energies")
            plt.axvline(x=0, color="red", linestyle="--", label="No binding")
            plt.axvline(x=-5.0, color="orange", linestyle="--", label="Strong binding cutoff")
            plt.legend()
            plt.grid(True, alpha=0.3)
            dist_plot_path = os.path.join(output_dir, f"energy_distribution_{timestamp}.png")
            plt.savefig(dist_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            report_files["energy_distribution"] = dist_plot_path

            # 4. Generate ranked bar plot (top 15)
            plt.figure(figsize=(12, 8))
            top_15 = df.head(15)
            y_pos = np.arange(len(top_15))
            plt.barh(y_pos, top_15["binding_energy_kcalmol"], color="skyblue", edgecolor="navy")
            plt.yticks(y_pos, top_15["compound_id"])
            plt.xlabel("Binding Energy (kcal/mol)")
            plt.title("Top 15 Compounds by Binding Energy")
            plt.axvline(x=0, color="red", linestyle="--", label="No binding")
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            ranked_plot_path = os.path.join(output_dir, f"ranked_barplot_{timestamp}.png")
            plt.savefig(ranked_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            report_files["ranked_barplot"] = ranked_plot_path

            # 5. Generate full report CSV
            full_report_path = os.path.join(output_dir, f"full_ranking_report_{timestamp}.csv")
            df.to_csv(full_report_path, index=False)
            report_files["full_report"] = full_report_path

            logger.info(f"Generated ranking report in {output_dir}")
            return report_files

        except Exception as e:
            logger.error(f"Error generating ranking report: {e}")
            return {}

    def compare_with_physics_model(
        self, ligand_pdbqt: str, receptor_pdbqt: str, df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Compare docking results with physics-based model (LJ + Coulomb).

        Args:
            ligand_pdbqt: Path to ligand PDBQT file (template)
            receptor_pdbqt: Path to receptor PDBQT file
            df: DataFrame with docking results (if None, uses self.results_data)

        Returns:
            Dictionary with comparison results
        """
        from sigma_hole_docking.docking_engine import SigmaHoleDockingEngine

        engine = SigmaHoleDockingEngine(use_physics_fallback=True)

        if df is None:
            if self.results_data is None:
                raise ValueError("No results data available. Load results first.")
            df = self.results_data.copy()
        else:
            df = df.copy()

        # Add physics-based scores
        physics_scores = []
        df["binding_energy_kcalmol"].values if "binding_energy_kcalmol" in df.columns else []

        logger.info("Calculating physics-based scores for comparison...")

        for idx, row in df.iterrows():
            # For this example, we'd need to generate the specific ligand file
            # In practice, you'd have the actual ligand files
            ligand_file = f"ligands/{row['compound_id']}_ligand.pdbqt"
            if os.path.exists(ligand_file):
                try:
                    physics_score, ok = engine.calculate_physics_score(ligand_file, receptor_pdbqt)
                    physics_scores.append(physics_score if ok else np.nan)
                except Exception as e:
                    logger.debug(f"Error calculating physics score for {row['compound_id']}: {e}")
                    physics_scores.append(np.nan)
            else:
                physics_scores.append(np.nan)

        df["physics_energy_kcalmol"] = physics_scores

        # Calculate correlation (excluding NaN values)
        valid_mask = ~(
            np.isnan(df["physics_energy_kcalmol"]) | np.isnan(df["binding_energy_kcalmol"])
        )
        if valid_mask.sum() > 2:
            correlation = np.corrcoef(
                df.loc[valid_mask, "binding_energy_kcalmol"],
                df.loc[valid_mask, "physics_energy_kcalmol"],
            )[0, 1]
        else:
            correlation = None

        comparison = {
            "compounds_with_physics": int(valid_mask.sum()),
            "correlation": float(correlation) if correlation is not None else None,
            "mean_difference": float(
                np.nanmean(df["physics_energy_kcalmol"] - df["binding_energy_kcalmol"])
            )
            if valid_mask.sum() > 0
            else None,
            "rmse": float(
                np.sqrt(
                    np.nanmean((df["physics_energy_kcalmol"] - df["binding_energy_kcalmol"]) ** 2)
                )
            )
            if valid_mask.sum() > 0
            else None,
        }

        if comparison["correlation"] is not None:
            compounds_with_physics = comparison["compounds_with_physics"]
            correlation_value = comparison["correlation"]
            logger_info = f"Physics model comparison: {compounds_with_physics} compounds, correlation = {correlation_value:.3f}"
            logger.info(logger_info)
        else:
            compounds_with_physics = comparison["compounds_with_physics"]
            logger_info = (
                f"Physics model comparison: {compounds_with_physics} compounds, no correlation"
            )
            logger.info(logger_info)

        return comparison

    def generate_interaction_visualizations(
        self,
        receptor_pdbqt: str,
        ligand_dir: str,
        output_dir: str = "visualizations",
        num_visualizations: int = 5,
    ) -> List[str]:
        """
        Generate 3D interaction visualizations for top hits using py3Dmol.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            ligand_dir: Directory containing ligand PDBQT files
            output_dir: Directory to save visualizations
            num_visualizations: Number of top hits to visualize

        Returns:
            List of paths to created visualization files
        """
        try:
            from interaction_visualizer import (
                create_visualizer_for_top_hits,
                SigmaHoleInteractionVisualizer,
            )
        except ImportError as e:
            logger.warning(f"Could not import interaction visualizer: {e}")
            logger.warning("3D visualization will be skipped. Install py3Dmol for this feature.")
            return []

        # Check if we have ranked results to visualize
        if self.results_data is None:
            logger.warning("No results data available. Run load_results and rank_compounds first.")
            return []

        # Use the top hits from ranked results
        top_hits = self.results_data.head(num_visualizations)

        try:
            SigmaHoleInteractionVisualizer()
            created_files = create_visualizer_for_top_hits(
                receptor_pdbqt=receptor_pdbqt,
                top_hits_df=top_hits,
                ligand_dir=ligand_dir,
                output_dir=output_dir,
                num_visualizations=num_visualizations,
            )
            logger.info(
                f"Generated {len(created_files)} interaction visualizations in {output_dir}"
            )
            return created_files
        except Exception as e:
            logger.error(f"Error generating interaction visualizations: {e}")
            return []

    def generate_publication_figures(
        self, df: Optional[pd.DataFrame] = None, output_dir: str = "figures"
    ) -> Dict[str, str]:
        """
        Generate publication-ready figures for sigma-hole analysis.

        Args:
            df: DataFrame with results (if None, uses ranked results)
            output_dir: Directory to save figures

        Returns:
            Dictionary mapping figure type to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if df is None:
            df = self.rank_compounds()
        else:
            df = self.rank_compounds(df)

        figures = {}

        try:
            # Figure 1: Energy distribution with cutoff regions
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(
                df["binding_energy_kcalmol"], bins=50, edgecolor="black", alpha=0.7, color="skyblue"
            )

            # Color regions by binding strength
            for i, p in enumerate(patches):
                bin_center = (bins[i] + bins[i + 1]) / 2
                if bin_center < -5.0:
                    p.set_facecolor("darkgreen")
                elif bin_center < -2.0:
                    p.set_facecolor("gold")
                elif bin_center < 0.0:
                    p.set_facecolor("lightcoral")
                else:
                    p.set_facecolor("lightgray")

            plt.xlabel("Binding Energy (kcal/mol)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("Distribution of Sigma-Hole Binding Energies", fontsize=14, fontweight="bold")
            plt.axvline(x=0, color="black", linestyle="-", linewidth=2, label="No binding")
            plt.axvline(
                x=-5.0, color="darkgreen", linestyle="--", linewidth=2, label="Strong binder cutoff"
            )
            plt.axvline(
                x=-2.0,
                color="goldenrod",
                linestyle="--",
                linewidth=2,
                label="Moderate binder cutoff",
            )
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fig1_path = os.path.join(output_dir, f"energy_distribution_{timestamp}.png")
            plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
            plt.close()
            figures["energy_distribution"] = fig1_path

            # Figure 2: Ranked binding energies (top 20)
            plt.figure(figsize=(12, 8))
            top_20 = df.head(20)
            y_pos = np.arange(len(top_20))
            colors = []
            for energy in top_20["binding_energy_kcalmol"]:
                if energy < -5.0:
                    colors.append("darkgreen")
                elif energy < -2.0:
                    colors.append("gold")
                elif energy < 0.0:
                    colors.append("lightcoral")
                else:
                    colors.append("lightgray")

            plt.barh(y_pos, top_20["binding_energy_kcalmol"], color=colors, edgecolor="black")
            plt.yticks(y_pos, top_20["compound_id"])
            plt.xlabel("Binding Energy (kcal/mol)", fontsize=12)
            plt.title("Top 20 Sigma-Hole Binders", fontsize=14, fontweight="bold")
            plt.axvline(x=0, color="black", linestyle="-", linewidth=2)
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()

            fig2_path = os.path.join(output_dir, f"ranked_binders_{timestamp}.png")
            plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
            plt.close()
            figures["ranked_binders"] = fig2_path

            # Figure 3: Correlation with halogen type (if available)
            if "halogen" in df.columns:
                plt.figure(figsize=(10, 6))
                halogens = df["halogen"].unique()
                halogen_data = [
                    df[df["halogen"] == h]["binding_energy_kcalmol"].values for h in halogens
                ]

                box_plot = plt.boxplot(halogen_data, labels=halogens, patch_artist=True)
                for patch in box_plot["boxes"]:
                    patch.set_facecolor("lightblue")

                plt.xlabel("Halogen", fontsize=12)
                plt.ylabel("Binding Energy (kcal/mol)", fontsize=12)
                plt.title(
                    "Sigma-Hole Binding Energy by Halogen Type", fontsize=14, fontweight="bold"
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                fig3_path = os.path.join(output_dir, f"halogen_dependence_{timestamp}.png")
                plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
                plt.close()
                figures["halogen_dependence"] = fig3_path

            logger.info(f"Generated {len(figures)} publication figures in {output_dir}")
            return figures

        except Exception as e:
            logger.error(f"Error generating publication figures: {e}")
            return {}


def example_usage():
    """Example usage of the results analyzer."""
    analyzer = SigmaHoleResultsAnalyzer()

    print("Sigma Hole Results Analyzer Example")
    print("=" * 40)

    # Create example results data
    example_data = {
        "compound_id": [
            "iodobenzene",
            "bromobenzene",
            "chlorobenzene",
            "fluorobenzene",
            "2-iodopyridine",
            "3-iodopyridine",
            "4-iodopyridine",
            "phenyl_iodide",
            "ortho_diiodobenzene",
            "meta_diiodobenzene",
        ],
        "halogen": ["I", "Br", "Cl", "F", "I", "I", "I", "I", "I", "I"],
        "binding_energy_kcalmol": [
            -5.2,
            -4.1,
            -2.8,
            -1.2,  # Strong to weak binders
            -4.8,
            -3.9,
            -3.1,  # Pyridine derivatives
            -5.5,
            -6.2,
            -4.9,  # Multi-iodo compounds
        ],
        "scoring_method": ["vinardo"] * 10,
    }

    df_example = pd.DataFrame(example_data)
    csv_path = "example_results.csv"
    df_example.to_csv(csv_path, index=False)
    print(f"Created example results file: {csv_path}")

    # Load and analyze results
    print("\n1. Loading results...")
    df_loaded = analyzer.load_results(csv_path)
    print(f"Loaded {len(df_loaded)} compounds")

    print("\n2. Ranking compounds...")
    df_ranked = analyzer.rank_compounds()
    print("Top 5 compounds:")
    print(df_ranked[["rank", "compound_id", "halogen", "binding_energy_kcalmol"]].head())

    print("\n3. Summary statistics...")
    stats = analyzer.calculate_summary_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n4. Top hits...")
    top_hits = analyzer.get_top_hits(n=5)
    print(top_hits[["compound_id", "halogen", "binding_energy_kcalmol"]])

    print("\n5. Generating report...")
    try:
        report_files = analyzer.generate_ranking_report(output_dir="example_reports")
        print(f"Generated {len(report_files)} report files:")
        for report_type, file_path in report_files.items():
            print(f"  {report_type}: {file_path}")
    except Exception as e:
        print(f"Report generation failed: {e}")

    # Cleanup
    try:
        os.remove(csv_path)
        import shutil

        if os.path.exists("example_reports"):
            shutil.rmtree("example_reports")
    except Exception:
        pass

    return analyzer


if __name__ == "__main__":
    example_usage()
