#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-3:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-29

source ~/minigrid/bin/activate

python3 $1/trainer.py --seed $SLURM_ARRAY_TASK_ID --algorithm PPO --phase TrainingTasks --config $1/configs/$3 --log_path $2