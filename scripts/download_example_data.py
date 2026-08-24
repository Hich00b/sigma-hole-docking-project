#!/usr/bin/env python3
"""Download example receptor data for the sigma-hole docking project.

The lysozyme receptor (160 KB PDBQT) was removed from git to keep the
repository lightweight.  This script downloads a lysozyme structure from
the RCSB PDB and prints instructions for preparing it with AutoDockTools.

Usage:
    python scripts/download_example_data.py
"""

import os
import sys
import urllib.request

PDB_ID = "1LZA"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def download_lysozyme() -> str:
    """Download lysozyme PDB from RCSB and return the local path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    dest = os.path.join(OUTPUT_DIR, f"{PDB_ID}_lysozyme.pdb")
    print(f"Downloading {PDB_ID} from RCSB...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def print_instructions(pdb_path: str) -> None:
    print("\n" + "=" * 60)
    print("To prepare the receptor PDBQT for docking:")
    print("=" * 60)
    print(f"1. Open {pdb_path} in AutoDockTools (ADT)")
    print("2. Remove water molecules and unwanted heteroatoms")
    print("3. Edit → Hydrogens → Add → All Hydrogens")
    print("4. Grid → Macromolecule → Choose molecule")
    print("5. Save as PDBQT: examples/lysozyme_receptor.pdbqt")
    print("\nAlternatively, use prepare_receptor4.py from AutoDockTools:")
    print(f"  prepare_receptor4.py -r {pdb_path} -o examples/lysozyme_receptor.pdbqt")
    print("\nOr use the sigma-hole pipeline directly with a PDB input:")
    print("  from sigma_hole_docking import SigmaHolePipeline")
    print(f"  pipeline = SigmaHolePipeline()")
    print(f"  pipeline.run_full_pipeline(")
    print(f"      input_csv='examples/example_input.csv',")
    print(f"      receptor_input='{pdb_path}',")
    print(f"      receptor_input_type='pdb')")
    print("=" * 60)


def main() -> int:
    try:
        pdb_path = download_lysozyme()
        print_instructions(pdb_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
