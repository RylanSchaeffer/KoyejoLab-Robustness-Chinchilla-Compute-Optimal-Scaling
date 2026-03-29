# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository accompanies the paper "Chinchilla Compute-Optimal Scaling of Language Models Is Surprisingly Robust." It re-analyzes the Chinchilla neural scaling laws by correcting model parameter counts and testing robustness to perturbations of those parameter counts. The fitting code is adapted from Epoch AI's [Chinchilla replication](https://github.com/epoch-research/analyzing-chinchilla/).

## Setup

```bash
conda create -n chinchilla_env python=3.11 -y && conda activate chinchilla_env
pip install autograd matplotlib pandas scipy seaborn
```

LaTeX rendering is used in all plots (`text.usetex = True`), so a working LaTeX installation with `amsmath`, `xcolor`, and `bm` packages is required.

## Running Notebooks

All scripts must be run from the **project root** (not from the notebook subdirectory), because imports use `src.*` and data paths are relative to the root:

```bash
python notebooks/00_assessing_parameters/00_assessing_parameters.py
python notebooks/01_epoch_research_fitting/01_epoch_research_fitting.py
python notebooks/02_robustness_analysis/02_robustness_analysis.py
python notebooks/03_robustness_schematic/03_robustness_schematic.py
```

Each notebook caches computed results (CSV files) in its own `data/` subdirectory. Set `refresh = True` at the top of a script to force recomputation. Notebook 02 is compute-intensive (4000 bootstrap iterations x many perturbation types, parallelized via `multiprocessing.Pool`).

## Architecture

- **`src/epoch_research_chinchilla_fit.py`** - Core scaling law math: log-sum-exp loss model, Huber loss objective, parameter transforms, compute-optimal allocation. Uses `autograd.numpy` for automatic differentiation (not standard numpy).
- **`src/analyze.py`** - Bootstrap fitting pipeline (`compute_chinchilla_fit_dataframes`), robustness analysis with parameter perturbations (additive constant, multiplicative constant, log-normal noise, systematic bias), data loading from CSV files.
- **`src/plot.py`** - Shared plotting setup (seaborn whitegrid, LaTeX fonts, `save_plot_with_multiple_extensions` saves both PDF and PNG).
- **`notebooks/`** - Numbered analysis scripts (00-03), each with `data/` (cached CSVs) and `results/` (figures) subdirectories.
- **`data/`** - Input data: `chinchilla_model_parameters.csv` (model architecture details) and `epoch_research_svg_extracted_data.csv` (training runs extracted from Chinchilla paper figures).

## Key Conventions

- Format code with [black](https://github.com/psf/black) before submitting PRs.
- The scaling law has 5 parameters: A, B, E, alpha, beta. These are fitted in log-transformed space and untransformed for display.
- `autograd` is used for gradient computation — `src/epoch_research_chinchilla_fit.py` imports `autograd.numpy` instead of `numpy`. Do not replace with standard numpy in that file.
