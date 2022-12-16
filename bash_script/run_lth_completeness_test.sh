#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/bash_run_lth_completeness_test_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/run_lth_completeness_test_$CURRENT.out
echo "Completeness score calculation using LTH"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7
# module load cuda/10.1
python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_test_completeness.py --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_mnist.yaml'> $slurm_output