import copy
import os
from multiprocessing import get_context
import torch as th
from baselines import vanilla, train_agent
from stable_baselines3 import PPO
from torch_to_subpolicies import nn_decomposition
from environment import basic_actions
from functions import trajectory_creation, get_option_program_length, get_top_k


th.multiprocessing.set_sharing_strategy("file_system")


def run_baselines(args_p, config, task_label, env, training_envs):
    name_postfix = f"{args_p.seed}_{config['clip_range']}_{config['ent_coef']}_{config['learning_rate']}"
    actions = env.action_space.n

    if args_p.baseline == "Vanilla":
        print("Task test - Vanilla")
        model_name = f"task{task_label}_Vanilla_seed" + name_postfix
        vanilla(env, model_name, args_p.seed, config, args_p.algorithm)
        return

    # Part two - Neural network Decomposition
    models = []
    for i in range(len(training_envs)):
        models.append(
            PPO.load(config["log_path"] + f"task{i + 1}_seed{args_p.seed}_MODEL")
        )

    program_stack = []
    for i in range(len(training_envs)):
        program_stack.append(nn_decomposition(models[i]))

    if args_p.baseline == "NeuralAugmented":
        model_name = f"task{len(training_envs) + 1}_NeuralAugmented_seed" + name_postfix
        selected_programs = [basic_actions(i) for i in range(actions)]
        selected_programs += [program_stack[i][0] for i in range(len(training_envs))]
        option_sizes = [1 for _ in range(len(selected_programs))]
        train_agent(
            env,
            selected_programs,
            option_sizes,
            model_name,
            args_p.seed,
            config,
            args_p.algorithm,
        )
        return

    # Part three - Program selection
    states_lst = []
    actions_lst = []

    for i, training_env in enumerate(training_envs):
        s, a = trajectory_creation(
            training_env, models[i], 1, deterministic=config["deterministic"]
        )
        states_lst.append(s)
        actions_lst.append(a)

    if args_p.baseline == "DecOptionsWhole":
        model_name = f"task{len(training_envs) + 1}_DecOptionsWhole_seed" + name_postfix
        selected_programs = [basic_actions(i) for i in range(actions)]
        option_sizes = [1 for _ in range(actions)]
        selected_programs += [program_stack[i][0] for i in range(len(training_envs))]
        for i in range(len(training_envs)):
            option_sizes.append(
                get_option_program_length(
                    states_lst, actions_lst, program_stack[i][0], i, actions
                )
            )
        train_agent(
            env,
            selected_programs,
            option_sizes,
            model_name,
            args_p.seed,
            config,
            args_p.algorithm,
        )
        return

    programs, option_sizes = get_top_k(
        20, states_lst, actions_lst, training_envs, program_stack, actions
    )

    if args_p.baseline == "DecOptions":
        model_name = f"task{len(training_envs) + 1}_DecOptions_seed" + name_postfix
        selected_programs = [basic_actions(i) for i in range(actions)] + programs
        option_sizes = [1 for _ in range(actions)] + option_sizes
        train_agent(
            env,
            selected_programs,
            option_sizes,
            model_name,
            args_p.seed,
            config,
            args_p.algorithm,
        )
        return


def test_tasks(
    args_p, hyperparameter_seach_space, config, task_number, env, training_envs
):
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
                            copy.deepcopy(training_envs),
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

    run_baselines(args_p, config, task_number, env, training_envs)
    return
