"""
Sigma Hole Docking Engine

Orchestrator for docking and scoring calculations that properly evaluate
dummy atom electrostatics for sigma-hole interactions.

The heavy lifting is split into three mixin modules:
  - ``scoring.py``           — LJ + Coulomb energy, charge scaling, physics score
  - ``alignment.py``         — molecular alignment along the C-X extension
  - ``pose_optimization.py`` — local rigid-body pose refinement

This module wires them together and provides the Vina/Smina subprocess
integration and batch-scoring entry points.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sigma_hole_docking.alignment import AlignmentMixin
from sigma_hole_docking.pdbqt_io import parse_pdbqt as _parse_pdbqt_file
from sigma_hole_docking.pose_optimization import PoseOptimizationMixin
from sigma_hole_docking.scoring import ScoringMixin

logger = logging.getLogger(__name__)


class SigmaHoleDockingEngine(ScoringMixin, AlignmentMixin, PoseOptimizationMixin):
    """Orchestrates sigma-hole docking: alignment, physics scoring, and Vina/Smina fallback.

    Public API (backward-compatible):
        - ``calculate_physics_score(ligand_pdbqt, receptor_pdbqt) -> Tuple[float, bool]``
        - ``dock_and_score(...)`` / ``score_only(...)`` / ``batch_score(...)``
    """

    def __init__(
        self,
        use_physics_fallback: bool = True,
        dielectric_coeff: float = 0.0,
        charge_scale: float = 1.0,
    ):
        """Initialize the docking engine.

        Args:
            use_physics_fallback: Whether to use physics-based scoring as fallback.
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

    def _parse_pdbqt(self, pdbqt_path: str) -> List[Dict]:
        """Parse PDBQT file using the consolidated :mod:`pdbqt_io` parser.

        Applies ``self.charge_scale`` to dummy-atom charges after parsing.
        """
        atoms = _parse_pdbqt_file(pdbqt_path)
        if self.charge_scale != 1.0:
            for atom in atoms:
                if atom["is_dummy"]:
                    atom["charge"] *= self.charge_scale
        return atoms

    # ------------------------------------------------------------------ #
    #  Vina / Smina subprocess integration
    # ------------------------------------------------------------------ #

    def _run_external_docking(
        self,
        executable: str,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        scoring: str = "vinardo",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Run Vina or Smina subprocess docking. Returns (affinity, output) or (None, error)."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pdbqt = os.path.join(temp_dir, "output.pdbqt")
                log_file = os.path.join(temp_dir, f"{executable}.log")

                # fmt: off
                cmd = [
                    executable, "--receptor", receptor_pdbqt,
                    "--ligand", ligand_pdbqt, "--out", output_pdbqt,
                    "--log", log_file, "--scoring", scoring,
                    "--exhaustiveness", str(exhaustiveness),
                    "--num_modes", str(num_modes),
                ]
                # fmt: on

                if center_x is not None and center_y is not None and center_z is not None:
                    # fmt: off
                    cmd.extend([
                        "--center_x", str(center_x),
                        "--center_y", str(center_y),
                        "--center_z", str(center_z),
                    ])
                    # fmt: on
                if size_x is not None and size_y is not None and size_z is not None:
                    # fmt: off
                    cmd.extend([
                        "--size_x", str(size_x),
                        "--size_y", str(size_y),
                        "--size_z", str(size_z),
                    ])
                    # fmt: on

                logger.debug(f"Running {executable} command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.error(f"{executable} failed: {result.stderr}")
                    return None, f"{executable} error: {result.stderr}"

                affinity = self._parse_vina_affinity(log_file)
                if affinity is not None:
                    logger.info(
                        f"{executable} docking completed. Best affinity: {affinity:.4f} kcal/mol"
                    )
                    return affinity, result.stdout
                logger.warning(f"Could not parse affinity from {executable} output")
                return None, result.stdout

        except subprocess.TimeoutExpired:
            logger.error(f"{executable} docking timed out after 120 seconds")
            return None, f"{executable} timeout"
        except FileNotFoundError:
            logger.error(f"{executable} not found. Install it or check PATH.")
            return None, f"{executable} executable not found"
        except Exception as e:
            logger.error(f"Error running {executable} docking: {e}", exc_info=True)
            return None, f"Docking error: {str(e)}"

    def run_vina_docking(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        scoring: str = "vinardo",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Run AutoDock Vina docking. Returns (affinity, output) or (None, error)."""
        # fmt: off
        return self._run_external_docking(
            "vina", receptor_pdbqt, ligand_pdbqt, scoring,
            exhaustiveness, num_modes,
            center_x, center_y, center_z,
            size_x, size_y, size_z,
        )
        # fmt: on

    def run_smina_docking(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        scoring: str = "vinardo",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Run Smina docking, falling back to Vina if Smina is unavailable."""
        try:
            subprocess.run(["smina", "--help"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Smina not found, falling back to Vina")
            # fmt: off
            return self.run_vina_docking(
                receptor_pdbqt, ligand_pdbqt, scoring,
                exhaustiveness, num_modes,
                center_x, center_y, center_z,
                size_x, size_y, size_z,
            )
            # fmt: on
        # fmt: off
        return self._run_external_docking(
            "smina", receptor_pdbqt, ligand_pdbqt, scoring,
            exhaustiveness, num_modes,
            center_x, center_y, center_z,
            size_x, size_y, size_z,
        )
        # fmt: on

    # ------------------------------------------------------------------ #
    #  Scoring entry points
    # ------------------------------------------------------------------ #

    def score_only(self, receptor_pdbqt: str, ligand_pdbqt: str, method: str = "auto") -> float:
        """Calculate binding energy using score_only mode.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file.
            ligand_pdbqt: Path to ligand PDBQT file.
            method: Scoring method ('auto', 'vinardo', 'ad4', 'vina', 'smina', 'physics').

        Returns:
            Binding energy in kcal/mol.
        """
        if method == "physics" or (method == "auto" and self.use_physics_fallback):
            logger.info("Using physics-based scoring")
            energy, ok = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            return energy if ok else 0.0

        if method in ["auto", "vinardo", "ad4", "vina"]:
            scoring = method if method != "auto" else "vinardo"
            affinity, _ = self.run_vina_docking(
                receptor_pdbqt, ligand_pdbqt, scoring=scoring, exhaustiveness=8, num_modes=1
            )
            if affinity is not None:
                return affinity
            logger.warning(f"Vina {scoring} scoring failed or returned None")

        if method in ["auto", "smina"]:
            affinity, _ = self.run_smina_docking(
                receptor_pdbqt, ligand_pdbqt, scoring="vinardo", exhaustiveness=8, num_modes=1
            )
            if affinity is not None:
                return affinity
            logger.warning("Smina scoring failed or returned None")

        if self.use_physics_fallback:
            logger.info("Falling back to physics-based scoring")
            energy, ok = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            return energy if ok else 0.0

        logger.error("All scoring methods failed")
        return 0.0

    def _record_docking_success(self, results: Dict, affinity: float, output: str) -> Dict:
        """Populate *results* on a successful Vina/Smina run and return it."""
        results["success"] = True
        results["best_affinity"] = affinity
        results["raw_output"] = output
        affinities = self._parse_all_affinities(output)
        results["all_affinities"] = affinities if affinities else [affinity]
        return results

    def dock_and_score(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        scoring: str = "vinardo",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
    ) -> Dict:
        """Perform docking and return comprehensive results.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file.
            ligand_pdbqt: Path to ligand PDBQT file.
            scoring: Scoring function to use.
            exhaustiveness: Exhaustiveness of search.
            num_modes: Number of binding modes to generate.
            center_x, center_y, center_z: Grid box center coordinates (Å).
            size_x, size_y, size_z: Grid box size dimensions (Å).

        Returns:
            Dictionary with docking results.
        """
        results: Dict = {
            "success": False,
            "best_affinity": None,
            "all_affinities": [],
            "binding_modes": [],
            "error": None,
        }

        if scoring == "physics":
            logger.info("Using physics-based scoring")
            physics_energy, ok = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
            results["success"] = ok
            results["best_affinity"] = physics_energy
            results["all_affinities"] = [physics_energy]
            results["method"] = "physics"
            return results

        try:
            # fmt: off
            docking_kwargs = dict(
                scoring=scoring, exhaustiveness=exhaustiveness, num_modes=num_modes,
                center_x=center_x, center_y=center_y, center_z=center_z,
                size_x=size_x, size_y=size_y, size_z=size_z,
            )
            # fmt: on

            # Try Vina first
            affinity, output = self.run_vina_docking(receptor_pdbqt, ligand_pdbqt, **docking_kwargs)
            if affinity is not None:
                return self._record_docking_success(results, affinity, output)

            # If Vina failed, try Smina
            logger.warning("Vina docking failed, trying Smina...")
            affinity, output = self.run_smina_docking(
                receptor_pdbqt, ligand_pdbqt, **docking_kwargs
            )
            if affinity is not None:
                return self._record_docking_success(results, affinity, output)

            # If both fail, use physics-based as last resort
            if self.use_physics_fallback:
                logger.warning("Both Vina and Smina failed, using physics-based scoring")
                physics_energy, ok = self.calculate_physics_score(ligand_pdbqt, receptor_pdbqt)
                results["success"] = ok
                results["best_affinity"] = physics_energy
                results["all_affinities"] = [physics_energy]
                results["method"] = "physics_fallback"
                return results

            results["error"] = "All docking/scoring methods failed"
            return results

        except Exception as e:
            logger.error(f"Error in dock_and_score: {e}")
            results["error"] = str(e)
            return results

    def batch_score(
        self,
        receptor_pdbqt: str,
        ligand_dir: str,
        output_csv: str,
        method: str = "auto",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
        size_x: Optional[float] = None,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
    ) -> pd.DataFrame:
        """Score multiple ligands against a single receptor with docking.

        Args:
            receptor_pdbqt: Path to receptor PDBQT file.
            ligand_dir: Directory containing ligand PDBQT files.
            output_csv: Path to save results CSV.
            method: Scoring method to use.
            exhaustiveness: Exhaustiveness of search.
            num_modes: Number of binding modes to generate.
            center_x, center_y, center_z: Grid box center coordinates (Å).
            size_x, size_y, size_z: Grid box size dimensions (Å).

        Returns:
            DataFrame with scoring results.
        """
        results = []

        ligand_files = [
            f
            for f in os.listdir(ligand_dir)
            if f.endswith(".pdbqt") and os.path.isfile(os.path.join(ligand_dir, f))
        ]
        logger.info(f"Found {len(ligand_files)} ligand files to score")

        # fmt: off
        docking_kwargs = dict(
            scoring=method, exhaustiveness=exhaustiveness, num_modes=num_modes,
            center_x=center_x, center_y=center_y, center_z=center_z,
            size_x=size_x, size_y=size_y, size_z=size_z,
        )
        # fmt: on

        for ligand_file in ligand_files:
            ligand_path = os.path.join(ligand_dir, ligand_file)
            ligand_name = ligand_file.replace("_ligand.pdbqt", "").replace(".pdbqt", "")

            try:
                docking_results = self.dock_and_score(
                    receptor_pdbqt=receptor_pdbqt,
                    ligand_pdbqt=ligand_path,
                    **docking_kwargs,
                )

                if docking_results["success"]:
                    affinity = docking_results["best_affinity"]
                else:
                    logger.warning(
                        f"Docking failed for {ligand_name}: "
                        f"{docking_results.get('error', 'Unknown error')}"
                    )
                    if self.use_physics_fallback:
                        affinity, ok = self.calculate_physics_score(ligand_path, receptor_pdbqt)
                        affinity = affinity if ok else float("nan")
                    else:
                        affinity = float("nan")

                if not np.isfinite(affinity):
                    logger.warning(f"Non-finite score for {ligand_name}: {affinity}")
                elif affinity == 0.0:
                    logger.warning(f"Score for {ligand_name} is exactly 0.0 — possible issue")

                results.append(
                    {
                        "compound_id": ligand_name,
                        "ligand_file": ligand_file,
                        "binding_energy_kcalmol": affinity,
                        "scoring_method": method,
                        "steric_clash": False,
                    }
                )

                if np.isfinite(affinity):
                    logger.debug(f"Scored {ligand_name}: {affinity:.4f} kcal/mol")

            except Exception as e:
                logger.error(f"Error scoring {ligand_name}: {e}")
                results.append(
                    {
                        "compound_id": ligand_name,
                        "ligand_file": ligand_file,
                        "binding_energy_kcalmol": float("nan"),
                        "scoring_method": method,
                        "error": str(e),
                        "steric_clash": False,
                    }
                )

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        logger.info(f"Saved scoring results to {output_csv}")

        return df_results
