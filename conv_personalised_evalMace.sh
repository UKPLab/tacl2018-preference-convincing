#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J cp_evalMACE
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./conv_personalised_evalMACE.err.%j
#SBATCH -o ./conv_personalised_evalMACE.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------

# enable this if running on lichtenberg
module load intel python/3.6.8

python3 python/analysis/habernal_comparison/personalised_tests.py 1
