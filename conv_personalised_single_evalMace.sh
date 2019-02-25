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

# ----------------------------------

module load python/3.6.2 
module load intel 

python3 python/analysis/habernal_comparison/personalised_tests.py 7
