#!/bin/bash

# Install non-Python dependencies
sudo apt-get update && sudo apt-get install -y wget texlive-latex-extra

# Initialize Git LFS
git lfs install

# Change to project directory
cd /workspaces/usaugment-experiments/

# Setup virtual environment
virtualenv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# Install the project as a package
pip install -e .

# Setup pre-commit hooks
pre-commit install
