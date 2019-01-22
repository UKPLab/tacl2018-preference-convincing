#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J sushi_100
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./sushi_100.err.%j
#SBATCH -o ./sushi_100.out.%j
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8182
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 python/analysis/sushi_10_tests.py
