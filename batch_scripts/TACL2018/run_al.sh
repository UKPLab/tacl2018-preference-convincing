#!/bin/sh

# Job name
#PBS -N conval

# Output file
#PBS -o conval_output.log

# Error file
#PBS -e conval_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=48GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch
source /home/es1595/.bashrc
source activate convincing

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/tacl2018-preference-convincing
export PYTHONPATH=$PYTHONPATH:"/work/es1595/tacl2018-preference-convincing/python"

#  run the script
python -u python/analysis/habernal_comparison/active_learning_tests.py
python -u python/analysis/habernal_comparison/compute_AL_metrics.py

# To submit: qsub <filename>.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
