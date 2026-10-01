# Linear System

This directory contains the Python code for the nonsymmetric linear-system experiment.

## Files

- `simple_linear3.py`: implementation of Anderson mixing and implicitly restarted Anderson mixing.
- `run_linear_fig1.py`: runs the four methods and generates Figure 1.
- `run_linear_table1.py`: generates the iteration-count comparison in Table 1.
- `results/`: numerical results and timing data.
- `figures/`: generated PDF and PNG figures.

## Problem settings

- System size: `1000 x 1000`
- Initial value: zero
- Stopping tolerance: `1e-6`
- Step size: `beta = 1e-3`
- The matrix and right-hand side are generated in the code.

## Compared methods in Figure 1

- AM-I with `m = 26`
- AM-II with `m = 26`
- IRAM-I with `(m,n) = (25,15)`
- IRAM-II with `(m,n) = (25,15)`

## Dependencies

The experiments were run with:

- Python 3.9.12
- NumPy 1.21.5
- SciPy 1.7.3
- Matplotlib 3.5.1

## Installation

Open a terminal in this `linear` directory and run:

```bash
python -m pip install -r ../requirements.txt
```

## Run

Keep the three Python files in this directory. Open a terminal here and run:

```bash
python run_linear_fig1.py
python run_linear_table1.py
```

Results are saved in `results/`, and figures are saved in `figures/`.
