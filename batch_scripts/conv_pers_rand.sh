#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J conv_rand
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./conv_pers_rand.err.%j
#SBATCH -o ./conv_pers_rand.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------

# enable this if running on lichtenberg
#module load intel python/3.6.8

# crowdGPPL
python3 python/analysis/habernal_comparison/random_selection_tests.py 0
python3 python/analysis/habernal_comparison/random_selection_tests.py 1

# GPPL
python3 python/analysis/habernal_comparison/random_selection_tests.py 6
python3 python/analysis/habernal_comparison/random_selection_tests.py 7

# crowdBT
python3 python/analysis/habernal_comparison/random_selection_tests.py 8
python3 python/analysis/habernal_comparison/random_selection_tests.py 9

# crowdBT -> GP
python3 python/analysis/habernal_comparison/random_selection_tests.py 10
python3 python/analysis/habernal_comparison/random_selection_tests.py 11
