#!/bin/bash
# Ensure this script is run from within ./models and all paths are relative to ./models

expected_dir="models"
if [ "$(basename "$(pwd)")" != "$expected_dir" ]; then
  echo "Please run this script from the './models' directory (e.g., 'cd models' first, then './download_vnncomp.sh')."
  exit 1
fi


echo $(pwd)
git clone https://github.com/VNN-COMP/vnncomp2025_benchmarks.git
cd vnncomp2025_benchmarks
./setup.sh
mv benchmarks/* ../vnnmodels/
cd ..
rm -rf vnncomp2025_benchmarks