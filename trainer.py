from environment_combo4 import get_training_tasks_combo4, get_test_task_combo4
from environment_minigrid import get_training_tasks_simplecross
from environment_minigrid import get_test_tasks_fourrooms
import argparse
from baselines import training_tasks_learning
import json
from baselines import vanilla
import copy
import os
from multiprocessing import get_context
import torch as th

th.multiprocessing.set_sharing_strategy("file_system")


def run_baselines(args_p, config, task_label, env):
    name_postfix = f"{args_p.seed}_{config['clip_range']}_{config['ent_coef']}_{config['learning_rate']}"
    if args_p.baseline == "Vanilla":
        print("Task test - Vanilla")
        model_name = f"task{task_label}_vanilla_seed" + name_postfix
        vanilla(env, model_name, args_p.seed, config, args_p.algorithm)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Training seed", default=0, type=int)
    parser.add_argument(
        "--algorithm", help="PPO or DQN", default="PPO", choices=["PPO", "DQN"]
    )
    parser.add_argument(
        "--phase",
        help="Training or Test task?",
        default="TestTasks",
        choices=["TrainingTasks", "TestTasks"],
    )
    parser.add_argument("--config", help="config file")
    parser.add_argument("--log_path", default="logs/")
    parser.add_argument("--baseline", default="Vanilla", choices=["Vanilla"])
    parser.add_argument("--parameter_sweep", default=False, type=bool)
    args_p = parser.parse_args()

    with open(args_p.config) as f:
        c = json.load(f)
        training_args = c["TrainingTasks"]
        config = c[str(args_p.phase)]
        hyperparameter_seach_space = c["search_space"]
        training_args["log_path"] = args_p.log_path
        config["log_path"] = args_p.log_path

    if config["task"] == "fourrooms":
        training_tasks = get_training_tasks_simplecross()
        env = get_test_tasks_fourrooms()[config["difficulty"]]
    elif config["task"] == "combogrid":
        training_tasks = get_training_tasks_combo4(config["size"])
        env = get_test_task_combo4([config["size"]])

    if args_p.phase == "TrainingTasks":
        training_tasks_learning(training_tasks, args_p.seed, training_args)
        return
    elif args_p.phase == "TestTasks":
        if args_p.parameter_sweep:
            clip_range = hyperparameter_seach_space["clip_range"]
            ent_coef = hyperparameter_seach_space["ent_coef"]
            learning_rate = hyperparameter_seach_space["learning_rate"]
            map_input = []
            for i in clip_range:
                for j in ent_coef:
                    for k in learning_rate:
                        new_config = copy.deepcopy(config)
                        new_config["clip_range"] = i
                        new_config["ent_coef"] = j
                        new_config["learning_rate"] = k
                        map_input.append(
                            (
                                copy.deepcopy(args_p),
                                new_config,
                                len(training_tasks) + 1,
                                copy.deepcopy(env),
                            )
                        )
            ncpus = min(
                int(os.environ.get("SLURM_CPUS_PER_TASK", default=1)),
                len(clip_range) * len(ent_coef) * len(learning_rate),
            )

            with get_context("spawn").Pool(processes=ncpus) as pool:
                list(pool.starmap(run_baselines, map_input))
                pool.close()
                pool.join()
            return
        else:
            run_baselines(args_p, config, len(training_tasks) + 1, env)
            return


if __name__ == "__main__":
    main()
