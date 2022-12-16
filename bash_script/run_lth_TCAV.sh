#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/bash_run_lth_TCAV_%j.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/bash_logs/run_lth_TCAV_$CURRENT.out
echo "Run TCAV"

#echo "cav vector sign (+ve) || dot product sign (+ve)"
##echo "cav vector sign (-ve) || dot product sign (-ve)"
#echo "-np.array(concept_model.coef_)"
#echo "np.dot(flatten_gradients, cav) > 0"

source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7
# module load cuda/10.1
python /ocean/projects/asc170022p/shg121/PhD/Project_Pruning/main_lth_tcav.py  --config '/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/config/BB_cub.yaml'> $slurm_output