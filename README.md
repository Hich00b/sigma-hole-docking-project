# Sigma-Hole Molecular Docking Pipeline

A Python package for modeling directional halogen-bonding (σ-hole) interactions in molecular docking. The pipeline places dummy-atom charge sites along C–X bonds, calibrates their charge from DFT-computed electrostatic-potential maxima (Vmax), and scores receptor–ligand interactions with Lennard-Jones + Coulomb physics.

## Overview

The sigma-hole pipeline models halogen bonding using:

1. **Dummy atoms (Extra Points)** — Virtual charge sites positioned along the C–X bond axis extension
2. **Vmax-based charges** — Dummy charge calibrated from DFT-computed electrostatic potential maxima via Multiwfn
3. **Physics-based scoring** — Lennard-Jones + Coulomb energy evaluation with directional corrections

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

## Prerequisites

**Python ≥ 3.9** with the following packages (installed automatically):
- numpy, pandas, rdkit, matplotlib, seaborn, py3Dmol

**Optional — AutoDock Vina / Smina** for empirical scoring:
```bash
conda install -c conda-forge autodock-vina
```
If Vina/Smina are not installed, the pipeline automatically falls back to physics-based scoring. No configuration change is needed.

## Quick Start

```python
from sigma_hole_docking import SigmaHolePipeline

pipeline = SigmaHolePipeline()

results = pipeline.run_full_pipeline(
    input_csv='examples/example_input.csv',
    receptor_input='examples/receptor.pdbqt',
    receptor_input_type='pdbqt',
)
```

The pipeline returns a dictionary with:
- `success`: Whether the pipeline completed
- `analysis`: Dictionary with `results_dataframe`, `top_hits`, `summary_statistics`
- `output_directory`: Path to output files (ranked CSV, reports)

## Input CSV Format

Your CSV file should contain at least these columns:

| Column | Description |
|--------|-------------|
| `compound_id` | Unique identifier for each compound |
| `smiles` | SMILES string (if generating structures from SMILES) |
| `halogen` | Halogen type (I, Br, Cl, F) |
| `vmax` | DFT Vmax value in kcal/mol |

Example:
```csv
compound_id,smiles,halogen,vmax
iodobenzene,c1ccccc1I,I,26.0
bromobenzene,c1ccccc1Br,Br,19.5
chlorobenzene,c1ccccc1Cl,Cl,14.2
```

## Receptor File

- Format: PDBQT (AutoDock format) or PDB
- PDBQT files can be prepared from PDB structures using [AutoDockTools](https://autodock.scripps.edu/)
- Example receptor: `examples/receptor.pdbqt` (small acetone test case)
- For a full lysozyme receptor, run `python scripts/download_example_data.py`

## Package Structure

```
sigma_hole_docking/
├── __init__.py            — Public API exports
├── pipeline.py            — Main pipeline orchestrator
├── charge_calculator.py   — Dummy charge from Vmax values
├── ligand_generator.py    — Ligand structures with dummy atoms
├── receptor_processor.py  — Receptor PDBQT preparation
├── docking_engine.py      — Docking orchestrator (< 500 lines)
├── scoring.py             — LJ + Coulomb energy scoring
├── alignment.py           — Molecular alignment along C-X axis
├── pose_optimization.py   — Local pose refinement
├── pdbqt_io.py            — Consolidated PDBQT parsing/writing
├── results_analyzer.py    — Results analysis and ranking
├── geometry_validator.py  — Input geometry validation
└── multiwfn_parser.py     — Multiwfn Vmax output parser
```

## Development

```bash
# Run tests
pytest --cov=sigma_hole_docking

# Lint
ruff check sigma_hole_docking tests

# Format
ruff format sigma_hole_docking tests
```

CI runs on Python 3.9–3.12 with ruff + pytest on every push and pull request.

## Example Data

- `examples/example_input.csv` — Sample compound input
- `examples/receptor.pdbqt` — Small acetone receptor for testing
- `examples/sigma_hole_docking_colab.ipynb` — Example notebook
- `scripts/download_example_data.py` — Downloads a lysozyme PDB from RCSB

## References

- Politzer, P., et al. (2013). Halogen bonding: An interaction divided. *CrystEngComm*, 15(16), 3029-3039.
- Cavallo, G., et al. (2016). The halogen bond. *Chemical Reviews*, 116(4), 2478-2601.
- Kolář, M. H., et al. (2019). σ-Hole interaction parameters. *J. Chem. Theory Comput.*, 15(5), 2972-2984.

## License

MIT License — see [LICENSE](LICENSE).

## Questions/Collaboration?

Contact: [Dr. Bensaada Hichem](http://bensaada.qzz.io/)
