#!/usr/bin/env bash
# Creates the "myenv" conda environment and registers it as a Jupyter kernel.
# Run from the submission/ folder:  bash setup_env.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n myenv python=3.13 -y
conda activate myenv
pip install -r requirements.txt
python -m ipykernel install --user --name myenv --display-name "myenv"
# Turn the .py files into notebooks (cells are marked with '# %%')
jupytext --to notebook q1_linear_regression.py q2_random_forest.py q3_automl.py
echo "Done. Activate with: conda activate myenv"
