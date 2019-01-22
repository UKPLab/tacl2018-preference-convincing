#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J synth_1
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e ./synth_1.err.%j
#SBATCH -o ./synth_1.out.%j
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8182
#SBATCH --exclusive

# ----------------------------------

module load python
module load intel

python3 python/analysis/simulations/synth_tests.py 1
