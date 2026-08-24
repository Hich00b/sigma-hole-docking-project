"""
Sigma Hole Charge Calculator

Calculates dummy atom charge from Vmax values using Coulomb's law for
sigma-hole (extra point) method in halogen bonding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SigmaHoleChargeCalculator:
    """
    Calculates dummy atom charges for sigma-hole modeling.

    Uses the formula: q = (Vmax × (r_iso − Δr)) / k_coulomb
    where:
        q = dummy charge (e)
        Vmax = ESP max (kcal/mol)
        r_iso = VdW radius / isosurface distance (Å)
        Δr = halogen-to-dummy distance (Å)
        k_coulomb = 332.06 kcal·Å/(mol·e²)
    """

    def __init__(self, k_coulomb: float = 332.06, charge_scale: float = 1.0):
        """
        Initialize the charge calculator.

        Args:
            k_coulomb: Coulomb constant (default: 332.06 kcal·Å/(mol·e²))
            charge_scale: Scaling factor for the calculated charge (default: 1.0)
        """
        self.k_coulomb = k_coulomb
        self.charge_scale = charge_scale

        # Default distances for halogen dummy atom placement
        # These are typical values: distance from halogen nucleus to dummy atom
        self.default_delta_r = {
            'F': 0.8,   # Fluorine
            'Cl': 1.0,  # Chlorine
            'Br': 1.1,  # Bromine
            'I': 1.2,   # Iodine
            'At': 1.3   # Astatine
        }

        # Typical VdW radii for determining isosurface distance
        self.vdw_radii = {
            'F': 1.47,
            'Cl': 1.75,
            'Br': 1.83,
            'I': 1.98,
            'At': 2.02
        }

    def calculate_charge(self, vmax: float, halogen: str,
                        delta_r: Optional[float] = None) -> float:
        """
        Calculate dummy atom charge from Vmax value using the correct physics formula:
        charge = Vmax * (r_iso - r_dummy) / k_coulomb

        Args:
            vmax: Electrostatic potential maximum (kcal/mol)
            halogen: Halogen element ('F', 'Cl', 'Br', 'I', 'At')
            delta_r: Distance from halogen nucleus to dummy atom (Å).
                    If None, uses default for the halogen.

        Returns:
            Dummy atom charge in electron units (e)
        """
        if delta_r is None:
            if halogen not in self.default_delta_r:
                raise ValueError(f"Unknown halogen: {halogen}. "
                               f"Supported: {list(self.default_delta_r.keys())}")
            delta_r = self.default_delta_r[halogen]

        # Get van der Waals radius for the halogen (approximate isosurface radius)
        if halogen not in self.vdw_radii:
            raise ValueError(f"Unknown halogen: {halogen}")
        r_iso = self.vdw_radii[halogen]  # Distance to Vmax isosurface

        # Calculate gap between dummy atom position and Vmax isosurface
        dist_gap = r_iso - delta_r
        if dist_gap <= 0:
            logger.warning(f"Dummy distance ({delta_r}) >= VdW radius ({r_iso}). "
                          f"Setting gap to 0.1 Å to avoid unphysical result.")
            dist_gap = 0.1

        # Correct formula: charge = Vmax * dist_gap / k_coulomb
        charge = (vmax * dist_gap) / self.k_coulomb
        charge *= self.charge_scale

        logger.debug(f"Calculated charge for {halogen}: "
                    f"Vmax={vmax} kcal/mol, r_iso={r_iso:.3f} Å, delta_r={delta_r:.3f} Å, "
                    f"dist_gap={dist_gap:.3f} Å → q={charge:.6f} e")

        # Validate the calculated charge
        vmax_calculated = self.k_coulomb * charge / dist_gap if dist_gap > 0 else 0.0
        validation_error = abs(vmax_calculated - vmax) / vmax * 100 if vmax != 0 else 0.0
        if validation_error > 1.0:  # More than 1% error
            logger.warning(f"Charge validation failed for {halogen}: "
                          f"Vmax_calculated={vmax_calculated:.2f} vs Vmax_input={vmax:.2f} "
                          f"(error: {validation_error:.1f}%)")
        else:
            logger.debug(f"Charge validation passed for {halogen}: "
                        f"Vmax_calculated={vmax_calculated:.2f} vs Vmax_input={vmax:.2f} "
                        f"(error: {validation_error:.1f}%)")

        return charge

    
    def batch_calculate_from_dataframe(self, df: pd.DataFrame,
                                     vmax_col: str = 'vmax',
                                     halogen_col: str = 'halogen',
                                     delta_r_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate charges for a batch of compounds in a DataFrame.

        Args:
            df: DataFrame containing Vmax and halogen information
            vmax_col: Column name for Vmax values
            halogen_col: Column name for halogen symbols
            delta_r_col: Optional column for custom delta_r values

        Returns:
            DataFrame with added 'dummy_charge' column
        """
        df = df.copy()

        if delta_r_col and delta_r_col in df.columns:
            # Use custom delta_r values
            df['dummy_charge'] = df.apply(
                lambda row: self.calculate_charge(
                    row[vmax_col],
                    row[halogen_col],
                    row[delta_r_col]
                ), axis=1
            )
        else:
            # Use default delta_r values
            df['dummy_charge'] = df.apply(
                lambda row: self.calculate_charge(
                    row[vmax_col],
                    row[halogen_col]
                ), axis=1
            )

        logger.info(f"Calculated charges for {len(df)} compounds")
        return df

    def save_charges(self, df: pd.DataFrame, output_path: str,
                    id_col: str = 'compound_id') -> None:
        """
        Save compound IDs and charges to CSV.

        Args:
            df: DataFrame with compound data and dummy_charge column
            output_path: Path to save CSV file
            id_col: Column name for compound identifiers
        """
        # Preserve all original columns and add/update the charge column
        output_df = df.copy()
        # Rename the dummy_charge column to dummy_charge_e for clarity
        if 'dummy_charge' in output_df.columns:
            output_df = output_df.rename(columns={'dummy_charge': 'dummy_charge_e'})
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved charges to {output_path}")

def example_usage():
    """Example usage of the charge calculator."""
    calculator = SigmaHoleChargeCalculator()

    # Example: Iodobenzene with Vmax = 26.0 kcal/mol
    vmax_iodobenzene = 26.0
    charge_i = calculator.calculate_charge(vmax_iodobenzene, 'I')
    print(f"Iodobenzene (Vmax={vmax_iodobenzene} kcal/mol): "
          f"dummy charge = {charge_i:.6f} e")

    
    # Batch example
    data = {
        'compound_id': ['iodobenzene', 'chlorobenzene', 'fluorobenzene'],
        'halogen': ['I', 'Cl', 'F'],
        'vmax': [26.0, 18.5, 12.0]  # Example Vmax values
    }
    df = pd.DataFrame(data)
    df_charged = calculator.batch_calculate_from_dataframe(df)
    print("\nBatch calculation:")
    print(df_charged[['compound_id', 'halogen', 'vmax', 'dummy_charge']])

    return charge_i

if __name__ == "__main__":
    example_usage()