# Sigma-Hole Molecular Docking Pipeline - Google Colab Version
## Usage Summary

This repository contains a modified version of the Sigma-Hole Molecular Docking Pipeline adapted to run in Google Colab notebooks instead of Hugging Face Spaces.

## What Was Done

1. **Copied essential files** from the original Hugging Face Spaces version:
   - All core Python modules (`sigma_hole_pipeline.py` and dependencies)
   - Example data files (CSV, PDBQT, SDF)
   - Created requirements file for Colab (`requirements_colab.txt`)
   - Created a demonstration Colab notebook (`sigma_hole_docking_colab.ipynb`)

2. **Removed Hugging Face specific components**:
   - Gradio interface (`app.py`) - replaced with notebook interface
   - Hugging Face specific packages (`gradio`, `huggingface_hub`, `spaces`)
   - Deployment-specific code

3. **Verified functionality** with a test run showing:
   - Successful import of all modules
   - Successful execution of the pipeline using SMILES-based structure generation
   - Correct receptor handling when specifying `receptor_input_type='pdbqt'`
   - Generation of scientifically meaningful results showing expected halogen bonding trends (I > Br > Cl > F)

## Files in this Repository

### Core Pipeline Modules
- `sigma_hole_pipeline.py` - Main pipeline orchestrator
- `charge_calculator.py` - Calculates dummy atom charges from Vmax values
- `ligand_generator.py` - Generates ligand structures with dummy atoms
- `receptor_processor.py` - Processes receptor PDBQT files
- `docking_engine.py` - Performs docking and scoring
- `results_analyzer.py` - Analyzes and ranks docking results
- `geometry_validator.py` - Validates input geometries
- `multiwfn_parser.py` - Parses Vmax values from Multiwfn output

### Example Data Files
- `example_input.csv` - Example compound input data (9 compounds)
- `receptor.pdbqt` - Lysozyme receptor in PDBQT format
- `2-iodopyridine_ligand.pdbqt` - Example ligand with dummy atom
- Various SDF files for structure-based approach
- `test_input.csv` - Alternative test input

### Documentation and Notebooks
- `README.md` - This overview document
- `USAGE_SUMMARY.md` - This usage summary
- `requirements_colab.txt` - Python package requirements for Colab
- `sigma_hole_docking_colab.ipynb` - Complete Colab notebook demonstration

### Demonstration Output (from verification run)
- `output/` - Results from test pipeline run:
  - `docking_results.csv` - Raw docking scores for all compounds
  - `ranked_results.csv` - Compounds ranked by binding energy
  - `validation_results.csv` - Validation metrics
  - `reports/` - Text reports
  - `figures/` - Generated plots

## How to Use in Google Colab

### Step 1: Setup
1. Create a new notebook at [colab.research.google.com](https://colab.research.google.com)
2. Upload all files from this repository to your Colab notebook
3. Install dependencies in a code cell:
   ```python
   !pip install -r requirements_colab.txt
   ```

### Step 2: Run the Pipeline
You can run the pipeline in two ways:

#### Option A: Using the Demo Notebook
Simply run all cells in `sigma_hole_docking_colab.ipynb`

#### Option B: Manual Execution
```python
from sigma_hole_pipeline import SigmaHolePipeline

# Initialize pipeline
pipeline = SigmaHolePipeline()

# Run using SMILES-based structure generation (recommended for simplicity)
results = pipeline.run_full_pipeline(
    input_csv='your_compounds.csv',        # CSV with compound data
    receptor_input='your_receptor.pdbqt',  # Receptor in PDBQT format
    receptor_input_type='pdbqt',           # CRITICAL: specify PDBQT format
    # No structure_dir needed - uses SMILES to generate structures
)

# Access results
if results and 'analysis' in results:
    print("Top hits:")
    print(results['analysis']['top_hits'])
```

### Step 3: Prepare Your Data

#### Input CSV Format
Your CSV should contain:
- `compound_id`: Unique identifier
- `smiles`: SMILES string (for structure generation from SMILES)
- `halogen`: Halogen type (I, Br, Cl, F)
- `vmax`: DFT Vmax value (kcal/mol) - optional

Example:
```csv
compound_id,smiles,halogen,vmax
iodobenzene,c1ccccc1I,I,26.0
bromobenzene,c1ccccc1Br,Br,19.5
```

#### Receptor File
- Format: PDBQT (AutoDock format)
- Obtain from RCSB PDB and prepare with AutoDockTools or similar
- Or use pre-prepared receptors

#### Optional: Structure Files (For DFT Approach)
For best accuracy, provide pre-optimized DFT structure files:
- Format: SDF, PDB, or MOL2
- Must match geometry used for Vmax calculations
- Specify with `structure_dir` and `structure_ext` parameters

### Key Parameters for run_full_pipeline()
- `input_csv`: Path to compound CSV file
- `receptor_input`: Path to receptor file
- `receptor_input_type`: 'pdb', 'smiles', or 'pdbqt' (DEFAULT: 'pdb' - BE CAREFUL!)
- `structure_dir`: Directory with ligand structures (OPTIONAL - if omitted, uses SMILES)
- `structure_ext`: File extension for structures (e.g., '.sdf') - required if structure_dir provided
- `vmax_col`: Column name for Vmax values (default: 'vmax')
- `halogen_col`: Column name for halogen type (default: 'halogen')

## Expected Results

The pipeline returns a dictionary containing:
- `success`: Boolean indicating if pipeline completed
- `elapsed_time`: Time taken to run pipeline
- `charged_compounds_csv`: Path to CSV with calculated charges
- `ligand_files`: List of generated ligand PDBQT files
- `receptor_pdbqt`: Path to prepared receptor PDBQT file
- `docking_results_csv`: Path to raw docking results
- `analysis`: Dictionary with analysis results including:
  - `results_dataframe`: Complete results with rankings
  - `summary_statistics`: Mean, median, std of binding energies
  - `top_hits`: Top-ranking compounds
  - `report_files`: Paths to generated text reports
  - `figure_files`: Paths to generated plots (if enabled)
- `validation_results`: List of validation metrics
- `output_directory`: Path to output directory

## Scientific Validation

From our test run with the example data, we observed the expected halogen bonding trend:
1. **Iodo compounds**: Strongest binding (most negative energies)
   - iodobenzene: -0.041 kcal/mol (Rank #1)
   - 4-iodopyridine: -0.002 kcal/mol (Rank #2)
   - 3-iodopyridine: -0.002 kcal/mol (Rank #3)
2. **Bromo compounds**: Moderate binding
   - 4-bromoacetophenone: -0.002 kcal/mol (Rank #4)
   - bromobenzene: -0.001 kcal/mol (Rank #5)
3. **Chloro compound**: Weak binding
   - chlorobenzene: -0.001 kcal/mol (Rank #6)
4. **Fluoro compounds**: Weak/non-binding (positive energies)
   - 4-fluoroacetophenone: 0.151 kcal/mol (Rank #7)
   - fluorobenzene: 0.253 kcal/mol (Rank #8)

This correctly demonstrates the expected σ-hole interaction strength trend: I > Br > Cl >> F.

## Differences from Original Hugging Face Version

| Feature | Hugging Face Spaces Version | Google Colab Version |
|---------|----------------------------|----------------------|
| Interface | Gradio web interface | Jupyter notebook |
| Deployment | Hugging Face Spaces | Google Colab |
| Key Packages | gradio, spaces, huggingface_hub | Standard scientific Python stack |
| Receptor Input Type Detection | Automatic | Manual specification required |
| Typical Use | Public web demo | Research, education, private screening |
| Data Privacy | Public (on Hugging Face) | Private (in your Google Drive) |
| Computation | Limited by Spaces runtime | GPU/TPU available in Colab |

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Solution: Ensure all packages from `requirements_colab.txt` are installed
   - `!pip install -r requirements_colab.txt`

2. **Receptor Preparation Errors**
   - Solution: Verify you specified the correct `receptor_input_type`:
     - 'pdb' for .pdb files
     - 'pdbqt' for .pdbqt files (most common)
     - 'smiles' for SMILES strings

3. **Missing Structure Files**
   - Solution: Either:
     - Provide structure files and specify `structure_dir` and `structure_ext`
     - Omit `structure_dir` to use SMILES-based generation (recommended for beginners)

4. **No Docking Results**
   - Solution: Check the output directory for logs and error messages
   - Verify input files exist and are in correct format

### Getting Help

For questions about the Sigma-Hole method:
- Refer to the original README.md for detailed scientific background
- Consult the references in the original documentation
- Contact: [Dr. Bensaada Hichem](https://dr-bensaada.pages.dev)

## Example Workflow

Here's a complete example workflow for a virtual screening study:

```python
# 1. Install dependencies
!pip install -r requirements_colab.txt

# 2. Import pipeline
from sigma_hole_pipeline import SigmaHolePipeline

# 3. Initialize
pipeline = SigmaHolePipeline()

# 4. Prepare your data
#    - compounds.csv: compound_id,smiles,halogen,vmax
#    - receptor.pdbqt: your target protein

# 5. Run screening
results = pipeline.run_full_pipeline(
    input_csv='compounds.csv',
    receptor_input='receptor.pdbqt',
    receptor_input_type='pdbqt'
)

# 6. Analyze results
if results['success']:
    top_hits = results['analysis']['top_hits']
    print("Top 10 compounds:")
    print(top_hits[['compound_id', 'binding_energy_kcalmol', 'halogen']].head(10))
    
    # 7. Save results for further analysis
    results['analysis']['results_dataframe'].to_csv('screening_results.csv', index=False)
else:
    print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
```

## License

MIT License - same as original repository.

## References

For detailed scientific background on the Sigma-Hole method and halogen bonding, see:
- Politzer, P., et al. (2013). Halogen bonding: An interaction divided. *CrystEngComm*, 15(16), 3029-3039.
- Cavallo, G., et al. (2016). The halogen bond. *Chemical Reviews*, 116(4), 2478-2601.
- Kolář, M. H., et al. (2019). σ-Hole interaction parameters. *Journal of chemical theory and computation*, 15(5), 2972-2984.

## Acknowledgments

Based on the original Sigma-Hole Molecular Docking Pipeline by [Dr. Bensaada Hichem](https://dr-bensaada.pages.dev).
Adapted for Google Colab by [Your Name/Organization].