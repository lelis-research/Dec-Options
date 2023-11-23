#!/bin/bash
#SBATCH --cpus-per-task=25   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10G        # memory per node
#SBATCH --time=0-$4:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-9

source ~/minigrid/bin/activate

python3 $1/trainer.py --seed $SLURM_ARRAY_TASK_ID --algorithm PPO --baseline $3 --phase TestTasks --parameter_sweep True --config $1/configs/fourrooms_easy.json --log_path $2