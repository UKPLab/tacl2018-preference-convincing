#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J conv_personalised
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./conv_personalised.err.%j
#SBATCH -o ./conv_personalised.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 python/analysis/habernal_comparison/personalised_tests.py 2
