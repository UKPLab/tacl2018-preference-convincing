#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J sushi_100
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./sushi_100.err.%j
#SBATCH -o ./sushi_100.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 python/analysis/sushi_100_test4.py
