#!/usr/bin/env python3
"""
Script to download example data for sigma-hole docking.

Downloads a lysozyme PDB from RCSB (PDB ID e.g. 1LZA) and prints 
instructions for AutoDockTools preparation.
"""

import os
import sys
import urllib.request
import gzip
import shutil


def download_pdb(pdb_id="1LZA", output_dir="examples"):
    """
    Download a PDB file from RCSB.
    
    Args:
        pdb_id: PDB identifier (default: 1LZA for lysozyme)
        output_dir: Directory to save the downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # RCSB PDB download URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_path = os.path.join(output_dir, f"{pdb_id}.pdb.gz")
    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    print(f"Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, gz_path)
        print(f"Downloaded to {gz_path}")
        
        # Extract the gzipped file
        print("Extracting PDB file...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(pdb_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the gzipped file to save space
        os.remove(gz_path)
        print(f"Extracted to {pdb_path}")
        print(f"File size: {os.path.getsize(pdb_path)} bytes")
        
    except Exception as e:
        print(f"Error downloading PDB {pdb_id}: {e}")
        return False
    
    return True


def print_preparation_instructions(pdb_path):
    """
    Print instructions for preparing the downloaded PDB with AutoDockTools.
    
    Args:
        pdb_path: Path to the downloaded PDB file
    """
    print("\n" + "="*60)
    print("PREPARATION INSTRUCTIONS FOR AUTODOCKTOOLS")
    print("="*60)
    print(f"1. Open {pdb_path} in AutoDockTools (ADT)")
    print("2. Remove water molecules and heteroatoms not needed for docking")
    print("3. Add polar hydrogens (if needed for your force field)")
    print("4. Add Kollman charges (for AutoDock Vina) or Gasteiger charges")
    print("5. Save as PDBQT format:")
    print("   - In ADT: Utilities → Write PDBQT")
    print("   - Or use prepare_receptor4.py from MGLTools:")
    print(f"     prepare_receptor4.py -r {pdb_path} -o receptor.pdbqt")
    print("\nFor lysozyme (1LZA), you may want to:")
    print("- Keep the protein chain(s) of interest")
    print("- Remove crystallographic water molecules beyond a certain distance")
    print("- Consider which residues form the active site for your study")
    print("="*60)


def main():
    """Main function."""
    print("Sigma-Hole Docking Example Data Downloader")
    print("=" * 50)
    
    # Download lysozyme PDB (1LZA is a common lysozyme structure)
    success = download_pdb("1LZA", "examples")
    
    if success:
        pdb_file = "examples/1LZA.pdb"
        if os.path.exists(pdb_file):
            print_preparation_instructions(pdb_file)
        else:
            print(f"Warning: Expected file {pdb_file} not found after download")
    else:
        print("Failed to download example data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
