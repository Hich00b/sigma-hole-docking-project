# Sigma-Hole Molecular Docking Pipeline - Google Colab Version

This is a modified version of the Sigma-Hole Molecular Docking Pipeline designed to run in Google Colab notebooks instead of Hugging Face Spaces.

## Overview

The Sigma-Hole pipeline models directional halogen-bonding (σ-hole) interactions using:
1. Dummy atoms (Extra Points) - Virtual charge sites positioned along the C–X bond axis
2. Vmax-based charges - Dummy charge calibrated from DFT-computed electrostatic potential maxima
3. Physics-based scoring - Lennard-Jones + Coulomb energy evaluation

## Files in this Repository

- `sigma_hole_pipeline.py` - Main pipeline orchestrator
- Supporting modules:
  - `charge_calculator.py` - Calculates dummy atom charges from Vmax values
  - `ligand_generator.py` - Generates ligand structures with dummy atoms
  - `receptor_processor.py` - Processes receptor PDBQT files
  - `docking_engine.py` - Performs docking and scoring
  - `results_analyzer.py` - Analyzes and ranks docking results
  - `geometry_validator.py` - Validates input geometries
  - `multiwfn_parser.py` - Parses Vmax values from Multiwfn output
- `sigma_hole_docking_colab.ipynb` - Example Colab notebook demonstrating usage
- `requirements_colab.txt` - Python package requirements for Colab
- Example data files:
  - `test_input.csv` - Example compound input data
  - `receptor.pdbqt` - Lysozyme receptor (default)
  - Various ligand and structure files (.pdbqt, .sdf)

## How to Use in Google Colab

1. **Create a new Colab notebook** at [colab.research.google.com](https://colab.research.google.com)

2. **Upload the files** from this repository to your Colab notebook:
   - All `.py` files
   - `requirements_colab.txt`
   - Example data files (or your own data)
   - `sigma_hole_docking_colab.ipynb` (optional - you can copy the code from here)

3. **Install dependencies** in a code cell:
   ```python
   !pip install -r requirements_colab.txt
   ```

4. **Run the pipeline** as demonstrated in the example notebook:
   ```python
   from sigma_hole_docking.pipeline import SigmaHolePipeline
   
   pipeline = SigmaHolePipeline()
   
   results = pipeline.run_full_pipeline(
       input_csv='your_compounds.csv',
       receptor_input='your_receptor.pdbqt',
       structure_dir='./your_structures',  # Optional: for DFT structures
       structure_ext='.sdf'                 # Optional: structure file extension
   )
   ```

5. **Analyze results** - The pipeline returns a dictionary containing:
   - `docking_results`: DataFrame with docking scores and poses
   - `analysis`: Dictionary with various analysis results including top hits
   - Other processing intermediates

## Data Requirements

### Input CSV Format
Your CSV file should contain at least these columns:
- `compound_id`: Unique identifier for each compound
- `smiles`: SMILES string (if generating structures from SMILES)
- `halogen`: Halogen type (I, Br, Cl, F)
- `vmax`: DFT Vmax value (kcal/mol) - optional if using approximate values

Example:
```csv
compound_id,smiles,halogen,vmax
iodobenzene,c1ccccc1I,I,26.0
bromobenzene,c1ccccc1Br,Br,19.5
chlorobenzene,c1ccccc1Cl,Cl,14.2
```

### Receptor File
- Format: PDBQT (AutoDock format)
- Can be obtained from:
  - [RCSB PDB](https://www.rcsb.org) → prepare with AutoDockTools or similar
  - Pre-prepared receptors in the `static/receptors/` directory of the original repo

### Structure Files (Optional but Recommended)
For best accuracy, provide pre-optimized DFT structure files:
- Format: SDF, PDB, or MOL2
- Should match the geometry used for Vmax calculations
- No hydrogen addition should be performed (preserves optimized geometry)

## Differences from Hugging Face Spaces Version

### Removed Components
- Gradio interface (`app.py`) - replaced with notebook interface
- Hugging Face specific packages (`gradio`, `huggingface_hub`, `spaces`)
- Deployment-specific code

### Kept Components
- All core docking functionality
- Charge calculation algorithms
- Ligand generation with dummy atoms
- Receptor processing
- Physics-based scoring
- Results analysis and visualization helpers
- Geometry validation tools

## Performance Notes

- In Colab, you have access to GPU/runtime resources (though this pipeline is primarily CPU-based)
- First-time rdkit installation may take a few minutes
- Large virtual screensings may be limited by Colab's runtime limits

## Troubleshooting

### Common Issues
1. **RDKit installation problems**: Try restarting the runtime and reinstalling
2. **Missing dependencies**: Ensure all packages from `requirements_colab.txt` are installed
3. **File path issues**: Use absolute paths or ensure files are in the current working directory
4. **Geometry mismatches**: For best results, use pre-optimized DFT structures that match your Vmax calculations

## References

For more information about the Sigma-Hole method and halogen bonding, see:
- Politzer, P., et al. (2013). Halogen bonding: An interaction divided. *CrystEngComm*, 15(16), 3029-3039.
- Cavallo, G., et al. (2016). The halogen bond. *Chemical Reviews*, 116(4), 2478-2601.
- Kolář, M. H., et al. (2019). σ-Hole interaction parameters. *Journal of chemical theory and computation*, 15(5), 2972-2984.

## License

MIT License - see original repository for details.

## Questions/Collaboration?

Contact: [Dr. Bensaada Hichem](http://bensaada.qzz.io/)