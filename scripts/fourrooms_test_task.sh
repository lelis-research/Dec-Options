#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-29

source ~/minigrid/bin/activate

python3 $1/trainer.py --seed $SLURM_ARRAY_TASK_ID --algorithm PPO --baseline $3 --phase TestTasks --config $1/configs/$4 --log_path $2 --clip_range $5 --ent_coef $6 --learning_rate $7