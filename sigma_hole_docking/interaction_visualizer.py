"""
Sigma Hole Interaction Visualizer Module

Contains functionality for generating 3D visualizations of sigma-hole interactions
using py3Dmol.
"""

from __future__ import annotations

import logging
import os
from typing import List

import pandas as pd

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

logger = logging.getLogger(__name__)


def _parse_pdbqt_for_visualization(pdbqt_path: str) -> tuple[list[dict], str]:
    """
    Parse a PDBQT file to extract atom information for visualization.

    Args:
        pdbqt_path: Path to the PDBQT file

    Returns:
        Tuple of (atoms_list, pdbqt_content_string)
    """
    atoms = []
    lines = []

    try:
        with open(pdbqt_path, "r") as f:
            for line in f:
                lines.append(line)
                if line.startswith(("ATOM", "HETATM")):
                    # Parse PDBQT format
                    # Format: ATOM      1  I   LIG B   1       0.000   0.000   0.000  0.00  0.00    0.100 I
                    parts = line.split()
                    if len(parts) >= 11:  # Minimum for ATOM record
                        try:
                            atom = {
                                "atom_id": int(parts[1]),
                                "element": parts[
                                    2
                                ].capitalize(),  # First letter uppercase, rest lowercase
                                "residue_name": parts[3],
                                "chain_id": parts[4],
                                "residue_seq": int(parts[5]),
                                "x": float(parts[6]),
                                "y": float(parts[7]),
                                "z": float(parts[8]),
                                "occupancy": float(parts[9]) if len(parts) > 9 else 0.0,
                                "temp_factor": float(parts[10]) if len(parts) > 10 else 0.0,
                            }
                            # Handle charge if present (column 11)
                            if len(parts) > 11:
                                atom["charge"] = float(parts[11])
                            # Handle atom type if present (column 12)
                            if len(parts) > 12:
                                atom["atom_type"] = parts[12]

                            atoms.append(atom)
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Could not parse PDBQT line: {line.strip()}. Error: {e}")
                            continue
    except FileNotFoundError:
        logger.error(f"PDBQT file not found: {pdbqt_path}")
        return [], ""
    except Exception as e:
        logger.error(f"Error reading PDBQT file {pdbqt_path}: {e}")
        return [], ""

    return atoms, "".join(lines)


def _get_atom_color(element: str) -> str:
    """Get CPK color for an element."""
    # Standard CPK colors
    cpk_colors = {
        "H": "#FFFFFF",  # White
        "C": "#909090",  # Gray
        "N": "#3050F8",  # Blue
        "O": "#FF0D0D",  # Red
        "S": "#FFFF30",  # Yellow
        "P": "#FF8000",  # Orange
        "F": "#90E050",  # Green
        "Cl": "#1FF01F",  # Green
        "Br": "#A62929",  # Brown
        "I": "#940094",  # Purple
        "At": "#940094",  # Purple
    }
    return cpk_colors.get(element.upper(), "#CCCCCC")  # Default to light gray


def create_visualizer_for_top_hits(
    receptor_pdbqt: str,
    top_hits_df: pd.DataFrame,
    ligand_dir: str,
    output_dir: str = "visualizations",
    num_visualizations: int = 5,
) -> List[str]:
    """
    Create 3D visualizations for top hits showing receptor-ligand interactions.

    Args:
        receptor_pdbqt: Path to receptor PDBQT file
        top_hits_df: DataFrame containing top hits (must have 'compound_id' column)
        ligand_dir: Directory containing ligand PDBQT files
        output_dir: Directory to save visualizations
        num_visualizations: Number of top hits to visualize

    Returns:
        List of paths to created visualization files (HTML)
    """
    if py3Dmol is None:
        logger.error("py3Dmol is not installed. Cannot generate visualizations.")
        return []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse receptor once
    logger.info(f"Parsing receptor PDBQT: {receptor_pdbqt}")
    receptor_atoms, receptor_pdbqt_content = _parse_pdbqt_for_visualization(receptor_pdbqt)
    if not receptor_atoms:
        logger.error(f"Failed to parse receptor PDBQT: {receptor_pdbqt}")
        return []

    # Limit to requested number of visualizations
    top_hits_to_visualize = top_hits_df.head(num_visualizations)

    created_files = []

    for idx, (_, hit) in enumerate(top_hits_to_visualize.iterrows()):
        compound_id = str(hit["compound_id"])  # Ensure it's a string
        ligand_filename = f"{compound_id}_ligand.pdbqt"  # Assuming standard naming
        ligand_path = os.path.join(ligand_dir, ligand_filename)

        # If the exact file doesn't exist, try to find any PDBQT file for this compound
        if not os.path.exists(ligand_path):
            # Look for any PDBQT file matching the compound ID
            try:
                ligand_files = [
                    f
                    for f in os.listdir(ligand_dir)
                    if f.startswith(f"{compound_id}_") and f.endswith(".pdbqt")
                ]
                if ligand_files:
                    ligand_path = os.path.join(ligand_dir, ligand_files[0])
                else:
                    logger.warning(
                        f"No ligand PDBQT file found for compound {compound_id} in {ligand_dir}"
                    )
                    continue
            except OSError as e:
                logger.warning(f"Error accessing ligand directory {ligand_dir}: {e}")
                continue

        logger.info(f"Parsing ligand PDBQT for {compound_id}: {ligand_path}")
        ligand_atoms, ligand_pdbqt_content = _parse_pdbqt_for_visualization(ligand_path)
        if not ligand_atoms:
            logger.warning(f"Failed to parse ligand PDBQT for {compound_id}: {ligand_path}")
            continue

        # Create py3Dmol view
        view = py3Dmol.view(width=800, height=600)

        # Add receptor as lines (lighter representation)
        view.addModel(receptor_pdbqt_content, "pdbqt")
        view.setStyle({"model": -1}, {"stick": {"radius": 0.1, "color": "lightgray"}})

        # Add ligand as sticks (more prominent)
        view.addModel(ligand_pdbqt_content, "pdbqt")
        view.setStyle({"model": 1}, {"stick": {"radius": 0.2, "colorscheme": "yellowCarbon"}})

        # Set the view to show both models
        view.zoomTo()

        # Save as HTML file using write_html method
        output_file = os.path.join(output_dir, f"{compound_id}_interaction.html")
        html_string = view.write_html()
        with open(output_file, "w") as f:
            f.write(html_string)
        created_files.append(output_file)

        logger.info(f"Created visualization for {compound_id}: {output_file}")

    logger.info(f"Generated {len(created_files)} interaction visualizations in {output_dir}")
    return created_files
