# Modified Bratu Problem

This directory contains the Python code for the modified Bratu experiment.

## Files

- `test_bratu.py`: implementation of Anderson mixing and implicitly restarted Anderson mixing.
- `run_bratu.py`: runs the four methods used in the numerical comparison.
- `plot_bratu.py`: generates the residual-versus-iteration figure.
- `../requirements.txt`: shared Python package requirements in the repository root.
- `results/`: numerical residual and timing data.
- `figures/`: generated PDF and PNG figures.

## Problem settings

- Grid size: `Nx = 200`, `Ny = 200`
- `alpha = 20`
- `lambda = 1`
- Initial value: zero
- Stopping tolerance: `1e-6`
- Step size: `beta = 6e-6`

## Compared methods

- AM-I with `m = 26` (`anderson1`)
- AM-II with `m = 26` (`anderson2`)
- IRAM-I with `(m,n) = (25,15)` (`iram`, `type2=False`)
- IRAM-II with `(m,n) = (25,15)` (`iram`, `type2=True`)

## Dependencies

The experiments were run with:

- Python 3.9.12
- NumPy 1.21.5
- SciPy 1.7.3
- Matplotlib 3.5.1

## Installation

Open a terminal in this `bratu` directory and run:

```bash
python -m pip install -r ../requirements.txt
```

## Run

```bash
python run_bratu.py
python plot_bratu.py
```

Results are saved in `results/`, and figures are saved in `figures/`.
