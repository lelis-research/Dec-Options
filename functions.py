import copy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, PPO
import numpy as np
import torch as th
import torch.nn as nn


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(
        self,
        verbose=0,
        env=None,
        eval_freq=1000,
        n_eval_episodes=10,
        log_path="",
        model_name="",
        num_worker=1,
    ):
        super(CustomCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.x_axis = []
        self.y_axis = []
        self.log_path = log_path
        self.model_name = model_name
        self.num_worker = num_worker

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.eval_freq == 0:
            reward_list = []
            self.x_axis.append(self.n_calls * self.num_worker)
            for _ in range(self.n_eval_episodes):
                done = False
                c_reward = 0
                steps = 0
                actions = np.zeros(self.env.action_space.n)
                o = self.env.reset()

                while done == False:
                    steps += 1
                    a, _ = self.model.predict(th.tensor(o))
                    actions[a] += 1
                    o, r, done, _ = self.env.step(int(a))
                    c_reward += r

                reward_list.append(c_reward)
            self.y_axis.append(np.array(reward_list).mean())
            np.save(self.log_path + self.model_name + "_x", np.array(self.x_axis))
            np.save(self.log_path + self.model_name + "_y", np.array(self.y_axis))
        return True

    def _on_training_end(self) -> None:
        np.save(self.log_path + self.model_name + "_x", np.array(self.x_axis))
        np.save(self.log_path + self.model_name + "_y", np.array(self.y_axis))


def initialize_dqn(
    env,
    num_worker=4,
    seed=0,
    policy_kwargs=None,
    learning_rate=0.01,
    gamma=0.99,
    batch_size=32,
    gradient_steps=1,
    target_update_interval=10000,
    tau=1.0,
    learning_starts=2000,
):
    th.manual_seed(seed)

    def vec():
        nonlocal env
        return copy.deepcopy(env)

    workers = make_vec_env(vec, n_envs=num_worker)
    model = DQN(
        "MlpPolicy",
        workers,
        policy_kwargs=policy_kwargs,
        gamma=gamma,
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        tau=tau,
        learning_rate=learning_rate,
        seed=seed,
        verbose=0,
        learning_starts=learning_starts,
    )

    model.policy.mlp_extractor.policy_net.apply(init_weights)
    model.policy.mlp_extractor.value_net.apply(init_weights)
    return model


def initialize_ppo(
    env,
    num_worker=4,
    seed=0,
    policy_kwargs=None,
    clip_range=0.2,
    learning_rate=0.01,
    gamma=0.99,
    ent_coef=0.1,
    gae_lambda=0.99,
    rollout_length=128,
    n_epochs=8,
):
    th.manual_seed(seed)

    def vec():
        nonlocal env
        return copy.deepcopy(env)

    workers = make_vec_env(vec, n_envs=num_worker)
    model = PPO(
        "MlpPolicy",
        workers,
        policy_kwargs=policy_kwargs,
        gamma=gamma,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        n_steps=rollout_length,
        seed=seed,
        verbose=0,
        clip_range=clip_range,
        n_epochs=n_epochs,
    )
    model.policy.mlp_extractor.policy_net.apply(init_weights)
    model.policy.mlp_extractor.value_net.apply(init_weights)
    return model


def learning(
    model,
    eval_env,
    num_worker,
    log_path="./logs/",
    model_name="",
    time_steps=500000,
    eval_freq=5000,
    n_eval_episodes=20,
):
    eval_callback = CustomCallback(
        verbose=0,
        env=eval_env,
        eval_freq=eval_freq,
        model_name=model_name,
        n_eval_episodes=n_eval_episodes,
        log_path=log_path,
        num_worker=num_worker,
    )
    model.learn(total_timesteps=time_steps * num_worker, callback=[eval_callback])
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight)
