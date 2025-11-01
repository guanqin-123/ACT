# Abstract Constraint Transformer (ACT)

[![Continuous Integration (CI) Status](https://github.com/doctormeeee/Abstract-Constraint-Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/doctormeeee/Abstract-Constraint-Transformer/actions/workflows/ci.yml)

An end-to-end neural network verification platform that supports refinement-based precision, diverse models, input formats, and specification types.

## Quick Start

```bash
# 0. Preparation
Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

conda env create -f environment.yml
conda activate act-py312 # Activate an environment (python-3.12)

# 1. Clone repository
git clone --recursive https://github.com/SVF-tools/ACT.git
cd ACT

# 2. Place Gurobi license (Required for MILP optimization)
# Place your gurobi.lic file in ./modules/gurobi/ directory
cp /path/to/your/gurobi.lic ./modules/gurobi/gurobi.lic

# 3. Run ACT native verification
cd ../act
python main.py --verifier act


## License

ACT is licensed under GNU Affero General Public License v3.0 (AGPL-3.0).


## Acknowledgements

This project was developed with the assistance of GitHub Copilot to enhance code readability and efficiency. AI-generated suggestions were reviewed and tested by the contributors before inclusion.