import copy
import os
from multiprocessing import get_context
import torch as th
from baselines import vanilla

th.multiprocessing.set_sharing_strategy("file_system")


def run_baselines(args_p, config, task_label, env):
    name_postfix = f"{args_p.seed}_{config['clip_range']}_{config['ent_coef']}_{config['learning_rate']}"
    if args_p.baseline == "Vanilla":
        print("Task test - Vanilla")
        model_name = f"task{task_label}_vanilla_seed" + name_postfix
        vanilla(env, model_name, args_p.seed, config, args_p.algorithm)
    return


def test_tasks(args_p, hyperparameter_seach_space, config, task_number, env):
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
                            task_number,
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
        run_baselines(args_p, config, task_number, env)
        return
