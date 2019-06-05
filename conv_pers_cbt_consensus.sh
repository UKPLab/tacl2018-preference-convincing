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

# crowd consensus

## crowdBT
#OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/personalised_tests.py 9

### personalised
### crowdBT
OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/personalised_tests.py 8
#
## crowdGPPL consensus
#OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/personalised_tests.py 1
## crowdGPPL personalised
#OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/personalised_tests.py 0
#
## GPPL consensus
#OMP_NUM_THREADS=36 python3 python/analysis/habernal_comparison/personalised_tests.py 7
