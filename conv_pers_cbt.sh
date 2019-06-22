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
module load intel python/3.6.8

### personalised
OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 1

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 5

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 10

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 12

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 15

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 17

OMP_NUM_THREADS=24 python3 -u python/analysis/habernal_comparison/personalised_tests.py 6 0 20






