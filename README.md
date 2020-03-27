# Sampling from the Complement of a Polyhedron:  An MCMC Algorithm for Data Augmentation

This repository contains the code for the paper.

## Getting started

- `all_miplib` contains MIPLIB instances that were considered.
- `src/notebooks` contains Jupyter notebooks for the knapsack experiments. 
- `src/generate_instances.py` contains the code to parse `.mps` files, load and
pre-process the optimization problems to generate instances, and convert them 
to `.pickle` that we can use.
- `src/mcmc_miplib_HR.py` contains the code to load `.pickle` files and run an
experiment.
 
## Running MIPLIB experiments

1. Run `generate_instances.py` after updating `pname` (directory where files 
are stored) to create instances of the MIPLIB. Default settings are to set `scale=1`
and generate `N=40` instances.
2. Run `mcmc_miplib_HR.py` after updating `p` and `r` (directory where results
are stored). The main function proceeds as follows:
* Generates feasible points using a HR sampler.
* Generates infeasible points using the SB sampler.
* Generates a test set of feasible and infeasible points using HR samplers.
* Trains KDE and GMM baselines using feasible points only and evaluates on test
data.
* Trains a GBT using feasible and infeasible points and evaluates on test data.
* Summarizes accuracy, TPR, FPR, precision metrics.
