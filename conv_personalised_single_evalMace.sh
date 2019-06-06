#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J cp_sin_eM
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./conv_personalised.err.%j
#SBATCH -o ./conv_personalised.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=28672
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------

# enable this if running on lichtenberg
module load intel python/3.6.8

python3 -u python/analysis/habernal_comparison/personalised_tests.py 7
