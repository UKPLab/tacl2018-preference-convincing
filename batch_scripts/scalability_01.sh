#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J 01scalability
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./scalability_01.err.%j
#SBATCH -o ./scalability_01.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 python/analysis/habernal_comparison/scalability_tests_personalised.py 0
