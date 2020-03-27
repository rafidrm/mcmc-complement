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
 
