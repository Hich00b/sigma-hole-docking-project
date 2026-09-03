"""
Sigma Hole Molecular Docking Pipeline

Main pipeline for quantum-calibrated dummy atom (extra point) method
for modeling halogen-bonding (sigma-hole) interactions.

Orchestrates:
1. Charge calculation from Vmax values
2. Ligand generation with dummy atoms
3. Receptor preparation
4. Docking/scoring with electrostatics-aware methods
5. Results analysis and ranking
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# Import our custom modules
from .charge_calculator import SigmaHoleChargeCalculator
from .docking_engine import SigmaHoleDockingEngine
from .ligand_generator import SigmaHoleLigandGenerator
from .receptor_processor import SigmaHoleReceptorProcessor
from .results_analyzer import SigmaHoleResultsAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sigma_hole_pipeline.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SigmaHolePipeline:
    """
    Main pipeline for sigma-hole molecular docking studies.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the sigma-hole pipeline.

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or self._default_config()
        self._setup_directories()

        # Initialize components
        self.charge_calculator = SigmaHoleChargeCalculator(
            charge_scale=self.config.get("charge_scale", 1.0)
        )
        self.ligand_generator = SigmaHoleLigandGenerator()
        self.receptor_processor = SigmaHoleReceptorProcessor()
        self.docking_engine = SigmaHoleDockingEngine(
            use_physics_fallback=self.config["use_physics_fallback"],
            dielectric_coeff=self.config.get("dielectric_coeff", 0.0),
            charge_scale=self.config.get("charge_scale", 1.0),
        )

        self.results_analyzer = SigmaHoleResultsAnalyzer()
        logger.info("Sigma Hole Pipeline initialized")

    def _default_config(self) -> dict:
        """Default configuration parameters."""
        return {
            # Directories
            "work_dir": "./sigma_hole_work",
            "ligand_dir": "./ligands",
            "receptor_dir": "./receptors",
            "output_dir": "./output",
            "temp_dir": "./temp",
            # Calculation parameters
            "default_delta_r": 1.2,  # Default distance from halogen to dummy atom (Å)
            "use_physics_fallback": True,  # Use physics-based scoring if Vina/Smina fails
            "scoring_method": "vinardo",  # Default scoring function
            "vina_exhaustiveness": 8,
            "vina_num_modes": 9,
            # Grid box parameters for docking
            "grid_center_x": None,  # X coordinate of grid box center (Å)
            "grid_center_y": None,  # Y coordinate of grid box center (Å)
            "grid_center_z": None,  # Z coordinate of grid box center (Å)
            "grid_size_x": None,  # Size of grid box in X dimension (Å)
            "grid_size_y": None,  # Size of grid box in Y dimension (Å)
            "grid_size_z": None,  # Size of grid box in Z dimension (Å)
            # Validation parameters
            "expected_xo_distance": (2.8, 3.5),  # Expected halogen-oxygen distance for sigma-hole
            "validation_enabled": True,
            # Output parameters
            "save_intermediates": True,
            "generate_report": True,
            "generate_figures": True,
            "generate_control": False,  # Generate control ligands without dummy atoms for comparison
            # Verbosity
            "dielectric_coeff": 0.0,  # Constant dielectric (epsilon_r = max(coeff, 1.0)). Use 0.0 for gas-phase (epsilon_r=1.0)
            "verbose": False,
        }

    def _setup_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.config["work_dir"],
            self.config["ligand_dir"],
            self.config["receptor_dir"],
            self.config["output_dir"],
            self.config["temp_dir"],
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def run_charge_calculation(
        self,
        input_csv: str,
        vmax_col: str = "vmax",
        halogen_col: str = "halogen",
        id_col: str = "compound_id",
        delta_r_col: str | None = None,
    ) -> str:
        """
        Step 1: Calculate dummy atom charges from Vmax values.

        Args:
            input_csv: Path to input CSV with Vmax and halogen data
            vmax_col: Column name for Vmax values
            halogen_col: Column name for halogen symbols
            id_col: Column name for compound identifiers
            delta_r_col: Optional column for custom delta_r values

        Returns:
            Path to CSV with calculated charges
        """
        logger.info("Step 1: Calculating dummy atom charges from Vmax values")

        # Load input data
        df_input = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df_input)} compounds from {input_csv}")

        # Calculate charges
        df_charged = self.charge_calculator.batch_calculate_from_dataframe(
            df_input, vmax_col=vmax_col, halogen_col=halogen_col, delta_r_col=delta_r_col
        )

        # Save charged compounds
        output_csv = os.path.join(self.config["work_dir"], "charged_compounds.csv")
        self.charge_calculator.save_charges(df_charged, output_csv, id_col=id_col)
        logger.info(f"Saved charged compounds to {output_csv}")

        return output_csv

    def prepare_ligands(
        self,
        charged_csv: str,
        smiles_col: str = "smiles",
        halogen_col: str = "halogen",
        charge_col: str = "dummy_charge_e",
        id_col: str = "compound_id",
        add_dummy: bool = True,
        structure_dir: str | None = None,
        structure_ext: str = ".sdf",
    ) -> list[str]:
        """
        Step 2: Generate ligand PDBQT files with optional dummy atoms.

        Args:
            charged_csv: Path to CSV with charged compounds
            smiles_col: Column name for SMILES strings
            halogen_col: Column name for halogen symbols
            charge_col: Column name for dummy atom charges
            id_col: Column name for compound identifiers
            add_dummy: Whether to add dummy atom (True for sigma-hole, False for control)

        Returns:
            List of generated ligand PDBQT file paths
        """
        if add_dummy:
            logger.info("Step 2: Generating ligand PDBQT files WITH dummy atoms (sigma-hole)")
        else:
            logger.info("Step 2: Generating ligand PDBQT files WITHOUT dummy atoms (control)")

        # Load charged compounds
        df_charged = pd.read_csv(charged_csv)
        logger.info(f"Loaded {len(df_charged)} charged compounds")

        # Generate ligands
        if structure_dir:
            logger.warning(
                "Structure directory provided — using pre-optimized DFT structures "
                "(no MMFF geometry optimization). Ensure Vmax values match these geometries."
            )
            ligand_files = self.ligand_generator.batch_generate_ligands_from_structures(
                df_charged,
                structure_dir=structure_dir,
                output_dir=self.config["ligand_dir"],
                halogen_col=halogen_col,
                charge_col=charge_col,
                id_col=id_col,
                structure_ext=structure_ext,
                add_dummy=add_dummy,
            )
        else:
            logger.warning(
                "Generating ligands from SMILES (MMFF geometry) — may not match "
                "DFT-optimized geometry where Vmax was measured."
            )
            ligand_files = self.ligand_generator.batch_generate_ligands(
                df_charged,
                output_dir=self.config["ligand_dir"],
                smiles_col=smiles_col,
                halogen_col=halogen_col,
                charge_col=charge_col,
                id_col=id_col,
                add_dummy=add_dummy,
            )

        logger.info(f"Generated {len(ligand_files)} ligand PDBQT files")
        return ligand_files

    def prepare_receptor(
        self, receptor_input: str, input_type: str = "pdb", receptor_id: str = "receptor"
    ) -> str:
        """
        Step 3: Prepare receptor PDBQT file from PDB, SMILES, or PDBQT.

        Args:
            receptor_input: Path to receptor PDB file, SMILES string, or PDBQT file
            input_type: Type of input ('pdb', 'smiles', or 'pdbqt')
            receptor_id: Identifier for the receptor

        Returns:
            Path to prepared receptor PDBQT file
        """
        logger.info("Step 3: Preparing receptor PDBQT file")

        if input_type == "pdb":
            receptor_path = os.path.join(
                self.config["receptor_dir"], f"{receptor_id}_receptor.pdbqt"
            )
            success = self.receptor_processor.prepare_receptor_from_pdb(
                receptor_input, receptor_path
            )
        elif input_type == "smiles":
            receptor_path = os.path.join(
                self.config["receptor_dir"], f"{receptor_id}_receptor.pdbqt"
            )
            success = self.receptor_processor.prepare_receptor_from_smiles(
                receptor_input, receptor_path
            )
        elif input_type == "pdbqt":
            # If input is already a PDBQT file, just copy it to the expected location
            receptor_path = os.path.join(
                self.config["receptor_dir"], f"{receptor_id}_receptor.pdbqt"
            )
            try:
                import shutil

                shutil.copy2(receptor_input, receptor_path)
                logger.info(f"Copied PDBQT file: {receptor_input} -> {receptor_path}")
                success = True
            except OSError as e:
                logger.error(f"Failed to copy PDBQT file: {e}")
                success = False
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        if not success:
            raise RuntimeError(f"Failed to prepare receptor from {input_type}: {receptor_input}")

        logger.info(f"Prepared receptor: {receptor_path}")
        return receptor_path

    def run_docking(self, receptor_pdbqt: str, scoring_method: str | None = None) -> str:
        """
        Step 4: Run docking/scoring for all ligands against the receptor.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            scoring_method: Scoring method to use (None uses config default)

        Returns:
            Path to CSV with docking results
        """
        logger.info("Step 4: Running docking/scoring")

        if scoring_method is None:
            scoring_method = self.config["scoring_method"]

        # Get all ligand files
        ligand_files = [
            f
            for f in os.listdir(self.config["ligand_dir"])
            if f.endswith(".pdbqt") and os.path.isfile(os.path.join(self.config["ligand_dir"], f))
        ]

        if not ligand_files:
            raise RuntimeError(f"No ligand files found in {self.config['ligand_dir']}")

        logger.info(f"Found {len(ligand_files)} ligand files to dock")

        # Score all ligands
        results_csv = os.path.join(self.config["output_dir"], "docking_results.csv")

        # Determine grid box parameters
        center_x = self.config.get("grid_center_x")
        center_y = self.config.get("grid_center_y")
        center_z = self.config.get("grid_center_z")
        size_x = self.config.get("grid_size_x")
        size_y = self.config.get("grid_size_y")
        size_z = self.config.get("grid_size_z")

        # If grid center not provided, compute it from the receptor
        if center_x is None or center_y is None or center_z is None:
            computed_center_x, computed_center_y, computed_center_z = (
                self.docking_engine.compute_receptor_center(receptor_pdbqt)
            )
            if center_x is None and computed_center_x is not None:
                center_x = computed_center_x
            if center_y is None and computed_center_y is not None:
                center_y = computed_center_y
            if center_z is None and computed_center_z is not None:
                center_z = computed_center_z

        # Set default grid size if not provided
        if size_x is None or size_y is None or size_z is None:
            default_size = 20.0  # Default grid size in Å
            if size_x is None:
                size_x = default_size
            if size_y is None:
                size_y = default_size
            if size_z is None:
                size_z = default_size

        self.docking_engine.batch_score(
            receptor_pdbqt=receptor_pdbqt,
            ligand_dir=self.config["ligand_dir"],
            output_csv=results_csv,
            method=scoring_method,
            exhaustiveness=self.config.get("vina_exhaustiveness", 8),
            num_modes=self.config.get("vina_num_modes", 9),
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )

        logger.info(f"Docking results saved to {results_csv}")
        return results_csv

    def analyze_results(self, results_csv: str) -> dict:
        """
        Step 5: Analyze and rank docking results.

        Args:
            results_csv: Path to docking results CSV

        Returns:
            Dictionary with analysis results and report file paths
        """
        logger.info("Step 5: Analyzing and ranking results")

        # Load and rank results
        self.results_analyzer.load_results(results_csv)
        df_ranked = self.results_analyzer.rank_compounds()

        # Calculate summary statistics
        stats = self.results_analyzer.calculate_summary_statistics()

        # Get top hits
        top_hits = self.results_analyzer.get_top_hits(n=10)

        # Generate reports if enabled
        report_files = {}
        if self.config["generate_report"]:
            logger.info("Generating ranking report...")
            report_files = self.results_analyzer.generate_ranking_report(
                df_ranked, output_dir=os.path.join(self.config["output_dir"], "reports")
            )

        # Generate figures if enabled
        figure_files = {}
        if self.config["generate_figures"]:
            logger.info("Generating publication figures...")
            figure_files = self.results_analyzer.generate_publication_figures(
                df_ranked, output_dir=os.path.join(self.config["output_dir"], "figures")
            )

        # Save ranked results
        ranked_csv = os.path.join(self.config["output_dir"], "ranked_results.csv")
        df_ranked.to_csv(ranked_csv, index=False)
        logger.info(f"Saved ranked results to {ranked_csv}")

        analysis_results = {
            "results_dataframe": df_ranked,
            "summary_statistics": stats,
            "top_hits": top_hits,
            "report_files": report_files,
            "figure_files": figure_files,
            "ranked_results_path": ranked_csv,
        }

        return analysis_results

    def validate_results(self, receptor_pdbqt: str, analysis_results: dict) -> list[dict]:
        """
        Step 6: Validate geometry of top hits.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            analysis_results: Results from analyze_results()

        Returns:
            List of validation results for top hits
        """
        if not self.config["validation_enabled"]:
            logger.info("Validation disabled in config")
            return []

        logger.info("Step 6: Validating geometry of top hits")

        df_ranked = analysis_results["results_dataframe"]
        top_n = min(5, len(df_ranked))  # Validate top 5 hits
        top_hits = df_ranked.head(top_n)

        validation_results = []

        for _, row in top_hits.iterrows():
            compound_id = row["compound_id"]
            ligand_file = os.path.join(self.config["ligand_dir"], f"{compound_id}_ligand.pdbqt")

            if not os.path.exists(ligand_file):
                logger.warning(f"Ligand file not found for {compound_id}: {ligand_file}")
                validation_results.append(
                    {"compound_id": compound_id, "valid": False, "error": "Ligand file not found"}
                )
                continue

            # Determine grid box parameters for validation (same logic as in run_docking)
            center_x = self.config.get("grid_center_x")
            center_y = self.config.get("grid_center_y")
            center_z = self.config.get("grid_center_z")
            size_x = self.config.get("grid_size_x")
            size_y = self.config.get("grid_size_y")
            size_z = self.config.get("grid_size_z")

            # If grid center not provided, compute it from the receptor
            if center_x is None or center_y is None or center_z is None:
                computed_center_x, computed_center_y, computed_center_z = (
                    self.docking_engine.compute_receptor_center(receptor_pdbqt)
                )
                if center_x is None and computed_center_x is not None:
                    center_x = computed_center_x
                if center_y is None and computed_center_y is not None:
                    center_y = computed_center_y
                if center_z is None and computed_center_z is not None:
                    center_z = computed_center_z

            # Set default grid size if not provided
            if size_x is None or size_y is None or size_z is None:
                default_size = 20.0  # Default grid size in Å
                if size_x is None:
                    size_x = default_size
                if size_y is None:
                    size_y = default_size
                if size_z is None:
                    size_z = default_size

            grid_box_center = (
                (center_x, center_y, center_z)
                if all(v is not None for v in [center_x, center_y, center_z])
                else None
            )
            grid_box_size = (
                (size_x, size_y, size_z)
                if all(v is not None for v in [size_x, size_y, size_z])
                else None
            )

            validation = self.results_analyzer.validate_geometry(
                ligand_pdbqt=ligand_file,
                receptor_pdbqt=receptor_pdbqt,
                halogen=row["halogen"],
                expected_distance_range=self.config["expected_xo_distance"],
                grid_box_center=grid_box_center,
                grid_box_size=grid_box_size,
            )
            validation["compound_id"] = compound_id
            validation_results.append(validation)

            logger.info(
                f"Validation for {compound_id}: {'PASS' if validation['valid'] else 'FAIL'}"
            )

        return validation_results

    def run_full_pipeline(
        self,
        input_csv: str,
        receptor_input: str,
        receptor_input_type: str = "pdb",
        receptor_id: str = "receptor",
        vmax_col: str = "vmax",
        halogen_col: str = "halogen",
        id_col: str = "compound_id",
        smiles_col: str = "smiles",
        scoring_method: str | None = None,
        structure_dir: str | None = None,
        structure_ext: str = ".sdf",
    ) -> dict:
        """
        Run the complete sigma-hole pipeline.

        Args:
            input_csv: Path to input CSV with compound data (SMILES, halogen, Vmax)
            receptor_input: Path to receptor PDB file or SMILES string
            receptor_input_type: Type of receptor input ('pdb' or 'smiles')
            receptor_id: Identifier for the receptor
            vmax_col: Column name for Vmax values in input CSV
            halogen_col: Column name for halogen symbols
            id_col: Column name for compound identifiers
            smiles_col: Column name for SMILES strings
            scoring_method: Scoring method to use (None uses config default)

        Returns:
            Dictionary with all pipeline results

        Note:
            If config['generate_control'] is True, control ligands without dummy atoms
            will be generated for comparison. Otherwise, sigma-hole ligands with dummy
            atoms will be generated.
        """
        start_time = datetime.now(timezone.utc)
        logger.info("Starting Sigma Hole Molecular Docking Pipeline")
        logger.info("=" * 60)

        try:
            # Step 1: Charge calculation
            charged_csv = self.run_charge_calculation(
                input_csv=input_csv, vmax_col=vmax_col, halogen_col=halogen_col, id_col=id_col
            )

            # Step 2: Ligand generation
            # If generate_control is True, we generate ligands WITHOUT dummy atoms (for control)
            # If generate_control is False (default), we generate ligands WITH dummy atoms (for sigma-hole)
            add_dummy = not self.config.get("generate_control", False)
            ligand_files = self.prepare_ligands(
                charged_csv=charged_csv,
                smiles_col=smiles_col,
                halogen_col=halogen_col,
                charge_col="dummy_charge_e",  # This is what charge_calculator saves as
                id_col=id_col,
                add_dummy=add_dummy,
                structure_dir=structure_dir,
                structure_ext=structure_ext,
            )

            # Step 3: Receptor preparation
            receptor_pdbqt = self.prepare_receptor(
                receptor_input=receptor_input,
                input_type=receptor_input_type,
                receptor_id=receptor_id,
            )

            # Step 4: Docking/scoring
            results_csv = self.run_docking(
                receptor_pdbqt=receptor_pdbqt, scoring_method=scoring_method
            )

            # Merge docking results with original compound data to preserve ligand information
            df_docking = pd.read_csv(results_csv)
            df_charged = pd.read_csv(charged_csv)
            # Determine which columns to merge (handle cases where smiles may not be present when using structure files)
            merge_columns = ["compound_id", "halogen", "vmax", "dummy_charge_e"]
            if "smiles" in df_charged.columns:
                merge_columns.insert(1, "smiles")  # Insert after compound_id
            # Merge on compound_id to add back original ligand information
            df_merged = pd.merge(
                df_docking, df_charged[merge_columns], on="compound_id", how="left"
            )
            # Save merged results back to the same file
            df_merged.to_csv(results_csv, index=False)

            # Step 5: Results analysis
            analysis_results = self.analyze_results(results_csv)

            # Step 6: Validation (optional)
            validation_results = self.validate_results(
                receptor_pdbqt=receptor_pdbqt, analysis_results=analysis_results
            )

            # Compile final results
            end_time = datetime.now(timezone.utc)
            elapsed_time = end_time - start_time

            # Save validation results if validation was performed and results exist
            if self.config["validation_enabled"] and validation_results:
                validation_csv_path = os.path.join(
                    self.config["output_dir"], "validation_results.csv"
                )
                validation_df = pd.DataFrame(validation_results)
                validation_df.to_csv(validation_csv_path, index=False)
                logger.info(f"Validation results saved to: {validation_csv_path}")

            # Generate interaction visualizations for top hits if figures are enabled
            if self.config["generate_figures"] and validation_results:
                logger.info("Generating interaction visualizations for validated top hits...")
                try:
                    # Generate interaction visualizations for validated compounds
                    viz_output_dir = os.path.join(
                        self.config["output_dir"], "interaction_visualizations"
                    )
                    viz_files = self.results_analyzer.generate_interaction_visualizations(
                        receptor_pdbqt=receptor_pdbqt,
                        ligand_dir=self.config["ligand_dir"],
                        output_dir=viz_output_dir,
                        num_visualizations=5,  # Visualize top 5 hits
                    )

                    if viz_files:
                        logger.info(
                            f"Generated {len(viz_files)} interaction visualizations in {viz_output_dir}"
                        )
                    else:
                        logger.warning("No interaction visualizations were generated")
                except (OSError, ValueError, RuntimeError) as e:
                    logger.error(f"Failed to generate interaction visualizations: {e}")

            pipeline_results = {
                "success": True,
                "elapsed_time": elapsed_time,
                "charged_compounds_csv": charged_csv,
                "ligand_files": ligand_files,
                "receptor_pdbqt": receptor_pdbqt,
                "docking_results_csv": results_csv,
                "analysis": analysis_results,
                "validation_results": validation_results,
                "output_directory": self.config["output_dir"],
            }

            logger.info("Pipeline completed successfully!")
            logger.info(f"Total elapsed time: {elapsed_time}")
            logger.info(f"Results saved to: {self.config['output_dir']}")

            return pipeline_results

        except Exception as e:
            logger.exception("Pipeline failed")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": datetime.now(timezone.utc) - start_time,
            }


def create_example_input() -> str:
    """Create an example input CSV for demonstration."""
    example_data = {
        "compound_id": [
            "iodobenzene",
            "bromobenzene",
            "chlorobenzene",
            "fluorobenzene",
            "2-iodopyridine",
            "3-iodopyridine",
            "4-iodopyridine",
            "4-fluoroacetophenone",
            "4-bromoacetophenone",
        ],
        "smiles": [
            "c1ccccc1I",
            "c1ccccc1Br",
            "c1ccccc1Cl",
            "c1ccccc1F",
            "c1ccncc1I",
            "c1cccnc1I",
            "c1cnccc1I",
            "CC(=O)c1ccc(F)cc1",
            "CC(=O)c1ccc(Br)cc1",
        ],
        "halogen": ["I", "Br", "Cl", "F", "I", "I", "I", "F", "Br"],
        "vmax": [
            26.0,
            19.5,
            14.2,
            9.8,
            24.5,
            23.8,
            22.1,
            11.2,
            18.9,
        ],  # Example Vmax values (kcal/mol)
    }

    df_example = pd.DataFrame(example_data)
    input_csv = "example_input.csv"
    df_example.to_csv(input_csv, index=False)
    logger.info(f"Created example input file: {input_csv}")
    return input_csv


def main():
    """Main function to run the sigma-hole pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Sigma Hole Molecular Docking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input arguments
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with compound data")
    parser.add_argument(
        "--receptor", "-r", required=True, help="Receptor file (PDB) or SMILES string"
    )
    parser.add_argument(
        "--receptor-type", choices=["pdb", "smiles"], default="pdb", help="Type of receptor input"
    )
    parser.add_argument("--receptor-id", default="receptor", help="Identifier for the receptor")

    # Column specifications
    parser.add_argument("--vmax-col", default="vmax", help="Column name for Vmax values")
    parser.add_argument("--halogen-col", default="halogen", help="Column name for halogen symbols")
    parser.add_argument(
        "--id-col", default="compound_id", help="Column name for compound identifiers"
    )
    parser.add_argument("--smiles-col", default="smiles", help="Column name for SMILES strings")

    # Pipeline options
    parser.add_argument(
        "--scoring-method",
        choices=["vinardo", "ad4", "vina", "smina", "physics"],
        default="vinardo",
        help="Scoring method to use",
    )
    parser.add_argument(
        "--no-physics-fallback", action="store_true", help="Disable physics-based scoring fallback"
    )
    parser.add_argument("--no-validation", action="store_true", help="Disable geometry validation")
    parser.add_argument("--no-report", action="store_true", help="Skip generating reports")
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip generating publication figures"
    )
    parser.add_argument(
        "--no-intermediates", action="store_true", help="Do not save intermediate files"
    )
    parser.add_argument(
        "--generate-control",
        action="store_true",
        help="Generate control ligands without dummy atoms for comparison",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="Exhaustiveness for docking search (default: 8)",
    )
    parser.add_argument(
        "--num-modes", type=int, default=9, help="Number of binding modes to generate (default: 9)"
    )
    # Grid box options
    parser.add_argument(
        "--grid-center-x",
        type=float,
        default=None,
        help="X coordinate of grid box center for docking (Å)",
    )
    parser.add_argument(
        "--grid-center-y",
        type=float,
        default=None,
        help="Y coordinate of grid box center for docking (Å)",
    )
    parser.add_argument(
        "--grid-center-z",
        type=float,
        default=None,
        help="Z coordinate of grid box center for docking (Å)",
    )
    parser.add_argument(
        "--grid-size-x", type=float, default=None, help="Size of grid box in X dimension (Å)"
    )
    parser.add_argument(
        "--grid-size-y", type=float, default=None, help="Size of grid box in Y dimension (Å)"
    )
    parser.add_argument(
        "--grid-size-z", type=float, default=None, help="Size of grid box in Z dimension (Å)"
    )

    # Other options
    parser.add_argument(
        "--output-dir", default="./sigma_hole_output", help="Output directory for results"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--create-example", action="store_true", help="Create example input file and exit"
    )

    args = parser.parse_args()

    if args.create_example:
        input_file = create_example_input()
        print(f"Example input file created: {input_file}")
        print("Edit this file with your compound data and run the pipeline again.")
        return 0

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration from arguments, then merge with defaults to ensure all keys are present
    arg_config = {
        "work_dir": os.path.join(args.output_dir, "work"),
        "ligand_dir": os.path.join(args.output_dir, "ligands"),
        "receptor_dir": os.path.join(args.output_dir, "receptors"),
        "output_dir": args.output_dir,
        "temp_dir": os.path.join(args.output_dir, "temp"),
        "use_physics_fallback": not args.no_physics_fallback,
        "scoring_method": args.scoring_method,
        "vina_exhaustiveness": args.exhaustiveness,
        "vina_num_modes": args.num_modes,
        "grid_center_x": args.grid_center_x,
        "grid_center_y": args.grid_center_y,
        "grid_center_z": args.grid_center_z,
        "grid_size_x": args.grid_size_x,
        "grid_size_y": args.grid_size_y,
        "grid_size_z": args.grid_size_z,
        "validation_enabled": not args.no_validation,
        "generate_report": not args.no_report,
        "generate_figures": not args.no_figures,
        "save_intermediates": not args.no_intermediates,
        "generate_control": args.generate_control,
        "verbose": args.verbose,
    }

    # Get default config and update with argument config to ensure all keys are present
    default_config = SigmaHolePipeline()._default_config()
    default_config.update(arg_config)
    config = default_config

    # Create and run pipeline
    pipeline = SigmaHolePipeline(config=config)

    # Run the pipeline
    results = pipeline.run_full_pipeline(
        input_csv=args.input,
        receptor_input=args.receptor,
        receptor_input_type=args.receptor_type,
        receptor_id=args.receptor_id,
        vmax_col=args.vmax_col,
        halogen_col=args.halogen_col,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
        scoring_method=args.scoring_method,
    )

    # Print summary
    if results["success"]:
        print("\n" + "=" * 60)
        print("SIGMA HOLE PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Elapsed time: {results['elapsed_time']}")
        print(f"Results directory: {results['output_directory']}")

        # Print top hits
        if "analysis" in results and "top_hits" in results["analysis"]:
            top_hits = results["analysis"]["top_hits"]
            print(f"\nTop {min(5, len(top_hits))} Hits:")
            print("-" * 50)
            for _, row in top_hits.head().iterrows():
                print(
                    f"{row['compound_id']:15} {row['halogen']:2} "
                    f"{row['binding_energy_kcalmol']:7.3f} kcal/mol"
                )

        # Print validation summary
        if "validation_results" in results:
            valid_count = sum(1 for v in results["validation_results"] if v.get("valid", False))
            total_count = len(results["validation_results"])
            print(f"\nGeometry Validation: {valid_count}/{total_count} passed")

        print(f"\nDetailed results available in: {results['output_directory']}")
        return 0
    else:
        print(f"\nPIPELINE FAILED: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
