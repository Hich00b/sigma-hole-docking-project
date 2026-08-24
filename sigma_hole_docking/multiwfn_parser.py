"""
Multiwfn Vmax Output Parser

Parses Multiwfn output files to extract sigma-hole Vmax values.
Handles the standard Multiwfn ESP analysis output format.
"""

import os
import re
import glob
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default regex patterns for Vmax extraction from Multiwfn output
_PATTERNS = [
    # Pattern 1: "    Maximum ESP:   -12.345     at (  x,   y,   z)"
    # or "MEP Maximum (kcal/mol):    12.345 at ( x, y, z)"
    re.compile(
        r'''(?:MEP\s+Maximum|Maximum\s+ESP|Max\s+ESP)[:\s]+([+-]?\d+\.?\d*)\s+(?:kcal/mol)?''',
        re.IGNORECASE
    ),
    # Pattern 2: "The maximal value is   12.345   a.u."
    re.compile(
        r'''(?:maximal|maximum)\s+value\s+is\s+([+-]?\d+\.?\d*)\s+(?:a\.u\.|kcal/mol)''',
        re.IGNORECASE
    ),
    # Pattern 3: for surface-based analysis
    re.compile(
        r'''surface\s+.*?\s+max.*?\s+([+-]?\d+\.?\d*)\s+kcal/mol''',
        re.IGNORECASE
    ),
    # Pattern 4: "positive value at point    12.345"
    re.compile(
        r'''positive.*?point\s+([+-]?\d+\.?\d*)''',
        re.IGNORECASE
    ),
]

# Patterns for coordinates (optional)
_COORD_PATTERN = re.compile(
    r'''at\s*\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)'''
)


class MultiwfnParser:
    """
    Parser for Multiwfn Vmax output files.

    Extracts electrostatic potential maximum (Vmax) values from Multiwfn
    output files generated during DFT workflow. Supports both whole-space
    and isosurface ESP analysis.
    """

    def __init__(self):
        """Initialize the parser."""
        self.results: List[Dict] = []
        self._pattern = _PATTERNS  # List of compiled patterns
        self._coord_pattern = _COORD_PATTERN

    def _parse_surfanalysis_txt(self, filepath: str) -> Optional[Dict]:
        """Parse a Multiwfn _surfanalysis.txt file.

        These files list surface maxima/minima in a tabular format.
        The row marked with '*' in the maxima section is the global
        surface maximum — the sigma-hole Vmax in kcal/mol.

        Returns dict with vmax, coords, unit, or None if no starred maximum.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Cannot read {filepath}: {e}")
            return None

        # Find the "Number of surface maxima:" section
        in_maxima = False
        past_header = False
        vmax_kcal = None
        coords = None

        for line in content.splitlines():
            stripped = line.strip()

            if 'Number of surface maxima' in stripped:
                in_maxima = True
                past_header = False
                continue

            if in_maxima:
                # Header line: "# a.u. eV kcal/mol X/Y/Z coordinate(Angstrom)"
                if stripped.startswith('#') or stripped == '':
                    past_header = True
                    continue
                if not past_header:
                    continue

                # Data line — may start with '*'
                is_starred = stripped.startswith('*')
                # Pattern: [*] <num> <a.u.> <eV> <kcal/mol> <x> <y> <z>
                parts = stripped.lstrip('*').split()
                if len(parts) < 7:
                    continue

                try:
                    kcal_val = float(parts[3])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                except (ValueError, IndexError):
                    continue

                if is_starred:
                    # This is the global maximum — the sigma-hole Vmax
                    vmax_kcal = kcal_val
                    coords = (x, y, z)
                    break  # No need to look further

                # Track the highest unstarred value as fallback
                if vmax_kcal is None or kcal_val > vmax_kcal:
                    vmax_kcal = kcal_val
                    coords = (x, y, z)

        if vmax_kcal is not None:
            logger.info(f"Surfanalysis Vmax = {vmax_kcal:.4f} kcal/mol from {filepath}")
            return {'vmax': vmax_kcal, 'unit': 'kcal/mol', 'coords': coords}

        return None

    def parse_vmax_output(self, multiwfn_file: str) -> Dict:
        """
        Extract Vmax from a single Multiwfn output file.

        Args:
            multiwfn_file: Path to Multiwfn output text file

        Returns:
            Dictionary with 'vmax', 'unit', 'coords', 'file', and 'success' keys
        """
        result = {
            'file': multiwfn_file,
            'vmax': None,
            'unit': 'kcal/mol',
            'coords': None,
            'success': False,
            'raw_match': None
        }

        if not os.path.exists(multiwfn_file):
            logger.error(f"File not found: {multiwfn_file}")
            return result

        try:
            # Try surfanalysis.txt format first (has '*' markers for sigma-hole Vmax)
            if multiwfn_file.endswith('_surfanalysis.txt') or 'surfanalysis' in os.path.basename(multiwfn_file):
                surf_result = self._parse_surfanalysis_txt(multiwfn_file)
                if surf_result is not None:
                    result['vmax'] = surf_result['vmax']
                    result['coords'] = surf_result.get('coords')
                    result['unit'] = surf_result.get('unit', 'kcal/mol')
                    result['success'] = True
                    result['raw_match'] = f"surfanalysis starred max: {surf_result['vmax']}"
                    logger.info(f"Extracted Vmax = {result['vmax']:.4f} from {multiwfn_file}")
                    return result

            with open(multiwfn_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            vmax = None

            # Try each pattern
            for pattern in self._pattern:
                match = pattern.search(content)
                if match:
                    try:
                        vmax = float(match.group(1))
                        result['raw_match'] = match.group(0)
                        break
                    except (ValueError, IndexError):
                        continue

            # Check if value is in a.u. and convert to kcal/mol (1 a.u. = 627.509 kcal/mol)
            if result['raw_match'] is not None and 'a.u.' in result['raw_match'].lower():
                if vmax is not None:
                    vmax_conv = vmax * 627.509
                    logger.debug(f"Converted Vmax from a.u. to kcal/mol: {vmax:.6f} -> {vmax_conv:.6f}")
                    vmax = vmax_conv

            # Extract coordinates if present
            if vmax is not None and match is not None:
                coord_match = self._coord_pattern.search(content, match.span()[1])
            elif vmax is not None:
                coord_match = self._coord_pattern.search(content)
            else:
                coord_match = None
                if coord_match:
                    try:
                        result['coords'] = tuple(float(coord_match.group(i)) for i in range(1, 4))
                    except (ValueError, IndexError):
                        pass

            result['vmax'] = vmax
            result['success'] = vmax is not None

            if vmax is None:
                # Try to find any number in the file that looks like an ESP value
                all_numbers = re.findall(r'([+-]?\d+\.\d+)', content)
                candidates = [float(n) for n in all_numbers if 5.0 <= abs(float(n)) <= 200.0]
                if candidates:
                    # Return the largest positive value as Vmax
                    positive = [n for n in candidates if n > 10.0]
                    if positive:
                        result['vmax'] = max(positive)
                        result['success'] = True
                        logger.warning(f"No standard Vmax pattern found; heuristically extracted: {result['vmax']}")

            if result['success']:
                logger.info(f"Extracted Vmax = {result['vmax']:.4f} from {multiwfn_file}")
            else:
                logger.warning(f"No Vmax found in {multiwfn_file}")

        except Exception as e:
            logger.error(f"Error parsing {multiwfn_file}: {e}")

        return result

    def parse_batch_vmax(self, directory: str, pattern: str = "*.out",
                          compound_col: str = 'compound_id') -> pd.DataFrame:
        """
        Extract Vmax from multiple Multiwfn output files in a directory.

        Args:
            directory: Directory containing output files
            pattern: Glob pattern for file matching (e.g., "*.out", "*_multiwfn.txt")
            compound_col: Column name for compound identifier (derived from filename)

        Returns:
            DataFrame with compound_id, vmax, unit, coords, file columns
        """
        self.results = []

        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return pd.DataFrame()

        # Find all matching files
        search_path = os.path.join(directory, pattern)
        files = sorted(glob.glob(search_path))
        logger.info(f"Found {len(files)} Multiwfn output files matching '{pattern}' in {directory}")

        for f in files:
            file_result = self.parse_vmax_output(f)
            # Extract compound_id from filename (strip directory and extension)
            basename = os.path.splitext(os.path.basename(f))[0]
            file_result['compound_id'] = basename
            self.results.append(file_result)

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert parsed results to a DataFrame.

        Returns:
            DataFrame with columns: compound_id, vmax, unit, coords, file, success
        """
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def merge_with_input(self, pipeline_csv: str, compound_id_col: str = 'compound_id',
                          vmax_col_new: str = 'vmax_multiwfn') -> pd.DataFrame:
        """
        Merge parsed Vmax values into an existing pipeline input CSV.

        Args:
            pipeline_csv: Path to pipeline input CSV
            compound_id_col: Column name for compound identifier
            vmax_col_new: Name of the new column for parsed Vmax values

        Returns:
            Merged DataFrame with original data plus parsed Vmax
        """
        df_results = self.to_dataframe()
        if df_results.empty:
            logger.warning("No results to merge — returning original CSV unchanged")
            return pd.read_csv(pipeline_csv)

        df_pipeline = pd.read_csv(pipeline_csv)

        # Prepare results for merge
        df_results_renamed = df_results[['compound_id', 'vmax', 'unit', 'coords']].copy()
        df_results_renamed = df_results_renamed.rename(columns={
            'vmax': vmax_col_new,
            'unit': f'{vmax_col_new}_unit'
        })

        # Merge on compound_id
        df_merged = pd.merge(
            df_pipeline,
            df_results_renamed,
            left_on=compound_id_col,
            right_on='compound_id',
            how='left'
        )

        # Drop duplicate compound_id column if it exists
        if 'compound_id_y' in df_merged.columns:
            df_merged = df_merged.drop(columns=['compound_id_y'])
            df_merged = df_merged.rename(columns={'compound_id_x': 'compound_id'})

        matched = df_merged[vmax_col_new].notna().sum()
        logger.info(f"Merged Vmax: {matched}/{len(df_merged)} compounds matched")

        return df_merged


def example_usage():
    """Example usage of the Multiwfn parser."""
    parser = MultiwfnParser()

    # Example: extract Vmax from a single file
    # result = parser.parse_vmax_output("gaussian_output/multiwfn_out.txt")
    # print(f"Vmax: {result['vmax']:.4f} {result['unit']}")

    # Example: batch parse and merge with pipeline CSV
    # df_results = parser.parse_batch_vmax("multiwfn_outputs", pattern="*.txt")
    # print(df_results)

    # Example: merge Vmax into existing input CSV
    # df_merged = parser.merge_with_input("input.csv", compound_id_col="compound_id")
    # df_merged.to_csv("input_with_vmax.csv", index=False)

    print("MultiwfnParser example — configure paths to use")
    return parser


if __name__ == "__main__":
    example_usage()