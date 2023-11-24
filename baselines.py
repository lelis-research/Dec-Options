import torch as th
from functions import initialize_dqn, initialize_ppo, learning
import copy
from environment import MasterEnv


def vanilla(env, model_name, seed, args, algorithm="PPO"):
    if algorithm == "DQN":
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=args["network"])
        model = initialize_dqn(
            env,
            num_worker=args["num_worker"],
            seed=seed,
            policy_kwargs=policy_kwargs,
            learning_rate=args["learning_rate"],
            gamma=args["gamma"],
            learning_starts=args["learning_starts"],
            batch_size=args["batch_size"],
            gradient_steps=args["gradient_steps"],
            target_update_interval=args["target_update_interval"],
            tau=args["tau"],
        )
    else:
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU, net_arch=dict(pi=args["network"], vf=args["vf"])
        )
        model = initialize_ppo(
            env,
            num_worker=args["num_worker"],
            seed=seed,
            policy_kwargs=policy_kwargs,
            clip_range=args["clip_range"],
            learning_rate=args["learning_rate"],
            gamma=args["gamma"],
            ent_coef=args["ent_coef"],
            gae_lambda=args["gae_lambda"],
            rollout_length=args["rollout_length"],
            n_epochs=args["n_epochs"],
        )

    eval_env = copy.deepcopy(env)
    model = learning(
        model,
        eval_env,
        num_worker=args["num_worker"],
        log_path=args["log_path"],
        model_name=model_name,
        time_steps=args["num_iterations"],
        eval_freq=args["eval_freq"],
        n_eval_episodes=args["n_eval_episodes"],
    )
    model.save(args["log_path"] + model_name + "_MODEL")


def training_tasks_learning(training_envs, seed, training_tasks_args):
    print(training_tasks_args)

    for i, env in enumerate(training_envs):
        print(f"Task {i + 1}")
        model_name = f"task{i + 1}_seed{seed}"
        vanilla(env, model_name, seed, training_tasks_args)


def train_agent(env, programs, option_sizes=None, model_name, seed, args, algorithm="PPO"):
    # set the environment
    master_env = MasterEnv(env=env, option_sizes=option_sizes, options=programs)

    if algorithm == "DQN":
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=args["network"])
        model = initialize_dqn(master_env, num_worker=args['num_worker'], seed=seed,
                                policy_kwargs=policy_kwargs, learning_rate=args['learning_rate'],
                                gamma=args['gamma'], learning_starts=args['learning_starts'],
                                batch_size=args["batch_size"], gradient_steps=args["gradient_steps"],
                                target_update_interval=args["target_update_interval"],
                                tau=args['tau'])
    else:
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=dict(pi=args["network"], vf=args["vf"]))
        model = initialize_ppo(master_env, num_worker=args['num_worker'], seed=seed,
                            policy_kwargs=policy_kwargs, clip_range=args['clip_range'],
                            learning_rate=args['learning_rate'], gamma=args['gamma'],
                            ent_coef=args['ent_coef'], gae_lambda=args['gae_lambda'],
                            rollout_length=args['rollout_length'], n_epochs=args['n_epochs'])

    eval_env = copy.deepcopy(master_env)
    model = learning(model, eval_env, num_worker=args['num_worker'],
                    log_path=args['log_path'], model_name=model_name,
                    time_steps=args['num_iterations'], eval_freq=args['eval_freq'],
                    n_eval_episodes=args['n_eval_episodes'])

    ## save model
    model.save(args['log_path'] + model_name + "_MASTER")