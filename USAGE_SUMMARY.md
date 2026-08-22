# Sigma-Hole Molecular Docking Pipeline — Usage Summary

## Installation

```bash
pip install sigma-hole-docking
```

For development:
```bash
git clone https://github.com/Hich00b/sigma-hole-docking-project.git
cd sigma-hole-docking-project
pip install -e ".[dev]"
```

## Package Structure

### Core Pipeline Modules
- `sigma_hole_docking/pipeline.py` — Main pipeline orchestrator
- `sigma_hole_docking/charge_calculator.py` — Dummy atom charges from Vmax values
- `sigma_hole_docking/ligand_generator.py` — Ligand structures with dummy atoms
- `sigma_hole_docking/receptor_processor.py` — Receptor PDBQT preparation
- `sigma_hole_docking/docking_engine.py` — Docking orchestrator
- `sigma_hole_docking/scoring.py` — LJ + Coulomb energy scoring
- `sigma_hole_docking/alignment.py` — Molecular alignment along C-X axis
- `sigma_hole_docking/pose_optimization.py` — Local pose refinement
- `sigma_hole_docking/pdbqt_io.py` — Consolidated PDBQT parsing/writing
- `sigma_hole_docking/results_analyzer.py` — Results analysis and ranking
- `sigma_hole_docking/geometry_validator.py` — Input geometry validation
- `sigma_hole_docking/multiwfn_parser.py` — Multiwfn Vmax output parser

### Example Data
- `examples/example_input.csv` — Example compound input data
- `examples/receptor.pdbqt` — Small acetone receptor for testing
- `examples/sigma_hole_docking_colab.ipynb` — Example notebook
- `scripts/download_example_data.py` — Downloads lysozyme PDB from RCSB

## How to Use

### Option A: Python Script

```python
from sigma_hole_docking import SigmaHolePipeline

pipeline = SigmaHolePipeline()

results = pipeline.run_full_pipeline(
    input_csv='examples/example_input.csv',
    receptor_input='examples/receptor.pdbqt',
    receptor_input_type='pdbqt',
)

if results['success']:
    print("Top hits:")
    print(results['analysis']['top_hits'])
```

### Option B: Jupyter Notebook
Run all cells in `examples/sigma_hole_docking_colab.ipynb`.

### Input CSV Format

| Column | Description |
|--------|-------------|
| `compound_id` | Unique identifier |
| `smiles` | SMILES string (for structure generation) |
| `halogen` | Halogen type (I, Br, Cl, F) |
| `vmax` | DFT Vmax value (kcal/mol) |

Example:
```csv
compound_id,smiles,halogen,vmax
iodobenzene,c1ccccc1I,I,26.0
bromobenzene,c1ccccc1Br,Br,19.5
```

### Receptor File
- Format: PDBQT (AutoDock format) or PDB
- Obtain from [RCSB PDB](https://www.rcsb.org) and prepare with AutoDockTools
- Or run `python scripts/download_example_data.py` for a lysozyme example

### Key Parameters for run_full_pipeline()
- `input_csv`: Path to compound CSV file
- `receptor_input`: Path to receptor file
- `receptor_input_type`: 'pdb', 'smiles', or 'pdbqt' (default: 'pdb')
- `structure_dir`: Directory with ligand structures (optional — if omitted, uses SMILES)
- `structure_ext`: File extension for structures (e.g., '.sdf')
- `vmax_col`: Column name for Vmax values (default: 'vmax')
- `halogen_col`: Column name for halogen type (default: 'halogen')

## Pipeline Output

The pipeline returns a dictionary containing:
- `success`: Boolean indicating if pipeline completed
- `elapsed_time`: Time taken to run pipeline
- `charged_compounds_csv`: Path to CSV with calculated charges
- `ligand_files`: List of generated ligand PDBQT files
- `receptor_pdbqt`: Path to prepared receptor PDBQT file
- `docking_results_csv`: Path to raw docking results
- `analysis`: Dictionary with:
  - `results_dataframe`: Complete results with rankings
  - `summary_statistics`: Mean, median, std of binding energies
  - `top_hits`: Top-ranking compounds
  - `report_files`: Paths to generated text reports
  - `figure_files`: Paths to generated plots (if enabled)
- `validation_results`: List of validation metrics
- `output_directory`: Path to output directory

## Scientific Validation

From a test run with the example data, the expected halogen bonding trend is observed:

1. **Iodo compounds**: Strongest binding (most negative energies)
2. **Bromo compounds**: Moderate binding
3. **Chloro compounds**: Weak binding
4. **Fluoro compounds**: Weak/non-binding (positive energies)

This correctly demonstrates the expected σ-hole interaction strength trend: I > Br > Cl >> F.

## Prerequisites

**AutoDock Vina / Smina** (optional):
```bash
conda install -c conda-forge autodock-vina
```
If Vina/Smina are not installed, the pipeline automatically falls back to physics-based scoring.

## Troubleshooting

1. **Import Errors** — Ensure the package is installed: `pip install sigma-hole-docking`
2. **Receptor Preparation Errors** — Verify `receptor_input_type` matches your file format
3. **Missing Structure Files** — Omit `structure_dir` to use SMILES-based generation
4. **No Docking Results** — Check the output directory for logs and error messages

## Example Workflow

```python
from sigma_hole_docking import SigmaHolePipeline

pipeline = SigmaHolePipeline()

results = pipeline.run_full_pipeline(
    input_csv='compounds.csv',
    receptor_input='receptor.pdbqt',
    receptor_input_type='pdbqt',
)

if results['success']:
    top_hits = results['analysis']['top_hits']
    print("Top 10 compounds:")
    print(top_hits[['compound_id', 'binding_energy_kcalmol', 'halogen']].head(10))

    results['analysis']['results_dataframe'].to_csv('screening_results.csv', index=False)
else:
    print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
```

## License

MIT License — see [LICENSE](LICENSE).

## References

- Politzer, P., et al. (2013). Halogen bonding: An interaction divided. *CrystEngComm*, 15(16), 3029-3039.
- Cavallo, G., et al. (2016). The halogen bond. *Chemical Reviews*, 116(4), 2478-2601.
- Kolář, M. H., et al. (2019). σ-Hole interaction parameters. *J. Chem. Theory Comput.*, 15(5), 2972-2984.

## Acknowledgments

Based on the original Sigma-Hole Molecular Docking Pipeline by [Dr. Bensaada Hichem](http://bensaada.qzz.io/).
