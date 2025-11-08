# Abstract Constraint Transformation (ACT)

An end-to-end neural network verification platform that supports refinement-based precision, diverse models, input formats, and specification types.

## Quick Start


## 0. Preparation
Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) and create running environment.

```
conda env create -f environment.yml    # Install required lib packages to run ACT
conda activate act-py312 # Activate an environment (python-3.12)  # Activate the environment 
```

## 1. Clone repository
```
git clone --recursive https://github.com/SVF-tools/ACT.git
cd ACT
```

## 2. Place Gurobi license (for MILP optimization)
```
cp /path/to/your/gurobi.lic ./modules/gurobi/gurobi.lic  # put gurobi.lic file in ./modules/gurobi/ directory
```

# 3. Run ACT phases
```
python -m act.pipeline --help
```

### License
ACT is licensed under GNU Affero General Public License v3.0 (AGPL-3.0).


### Acknowledgements
This project was developed with the assistance of GitHub Copilot to enhance code readability and efficiency. AI-generated suggestions were reviewed and tested by the contributors before inclusion.
