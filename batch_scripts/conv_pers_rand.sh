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

# GPPL
OMP_NUM_THREADS=24 python3 python/analysis/habernal_comparison/random_selection_tests.py 6
OMP_NUM_THREADS=24 python3 python/analysis/habernal_comparison/random_selection_tests.py 7

# crowdGPPL
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 0
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 1

# crowdBT
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 8
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 9

# crowdBT -> GP
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 10
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/random_selection_tests.py 11
