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
#SBATCH -C avx

# ----------------------------------

module load intel python/3.6.8


python3 -u python/analysis/sushi_10_tests.py 0
python3 -u python/analysis/sushi_10_tests.py 2
python3 -u python/analysis/sushi_10_tests.py 4
