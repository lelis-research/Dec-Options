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
            if (self.n_calls - 1) % (self.eval_freq * 10) == 0:
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
    model.policy.action_net.apply(init_weights)
    model.policy.mlp_extractor.value_net.apply(init_weights)
    model.policy.value_net.apply(init_weights)
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
    model.policy.action_net.apply(init_weights)
    model.policy.mlp_extractor.value_net.apply(init_weights)
    model.policy.value_net.apply(init_weights)
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
        th.nn.init.xavier_normal_(m.weight)


def trajectory_creation(env, model, no, deterministic):
    actions = []
    states = []
    for _ in range(no):
        done = False
        o = env.reset()
        while done == False:
            a, _ = model.predict(th.tensor(o), deterministic=deterministic)
            states.append(o)
            actions.append(int(a))
            o, _, done, _ = env.step(int(a))
    return (np.array(states), np.array(actions))


def binary_seq(states, actions, program):
    bin_seq = np.zeros(len(states), dtype=int)
    for i, state in enumerate(states):
        if (
            program.predict_hierarchical(th.tensor(state).float(), epsilon=0.0)
            == actions[i]
        ):
            bin_seq[i] = 1
    return bin_seq


def get_option_length(no_actions, binary_sequences):
    """find best option length for each program"""
    value = np.inf
    o_l = None
    max_length = len(max(binary_sequences, key=len))
    for option_length in range(2, max_length):
        total_expected_nodes = 0
        break_flag = 0
        for binary_sequence in binary_sequences:
            counter = []
            h = 0
            for i in binary_sequence:
                if i == 1:
                    h += 1
                else:
                    if h >= option_length:
                        counter.append(h)
                    h = 0
            if h >= option_length:
                counter.append(h)

            length = 0
            for i in counter:
                length += int(i / option_length)

            if length == 0:
                break_flag += 1

            length = len(binary_sequence) + length * (1 - option_length)
            total_expected_nodes += len(binary_sequence) * (no_actions**length)

        if total_expected_nodes <= value:
            value = total_expected_nodes
            o_l = option_length

        if break_flag == len(binary_sequences):
            break

    return o_l


def get_option_program_length(states, actions, program, task_no, base_actions=3):
    new_states = copy.deepcopy(states)
    new_actions = copy.deepcopy(actions)

    binary_sequences = []
    for i, value in enumerate(new_states):
        if i == task_no:
            binary_sequences.append(np.zeros(len(actions[i])))
            continue
        sequence = binary_seq(value, new_actions[i], program)
        binary_sequences.append(sequence)
    return get_option_length(base_actions + 1, binary_sequences)


def expected_nodes(no_actions, binary_sequences_lst):
    no_actions = no_actions + len(binary_sequences_lst)
    # find best option length for each program
    option_lengths = []
    for binary_sequences in binary_sequences_lst:
        option_lengths.append(get_option_length(no_actions, binary_sequences))

    # check for the best cover
    table = []  # -> save d
    no_trajectories = len(binary_sequences_lst[0])
    no_options = len(option_lengths)
    for i in range(no_trajectories):
        no_steps = len(binary_sequences_lst[0][i])
        table.append(np.full(no_steps + 1, np.inf))
        table[i][0] = 0
        for j in range(no_steps):
            # checking trajectory i, step j -> update table[i][j]
            for k in range(no_options):
                counter = 0
                for p in range(j, min(j + option_lengths[k], no_steps)):
                    if binary_sequences_lst[k][i][p] == 1:
                        counter += 1
                if counter == option_lengths[k]:
                    table[i][j + option_lengths[k]] = min(
                        table[i][j + option_lengths[k]], table[i][j] + 1
                    )
            table[i][j + 1] = min(table[i][j + 1], table[i][j] + 1)

    total_expected_nodes = 0
    for i in table:
        total_expected_nodes += (len(i) - 1) * (int(no_actions) ** int(i[-1]))

    return (total_expected_nodes, option_lengths)


def one_leave_out_val(states, actions, programs, task_no, base_actions=3):
    new_states = copy.deepcopy(states)
    new_actions = copy.deepcopy(actions)

    binary_sequences_lst = []
    for program in programs:
        binary_sequences = []
        for i, new_state in enumerate(new_states):
            if i == task_no:
                binary_sequences.append(np.zeros(len(actions[i])))
                continue
            sequence = binary_seq(new_state, new_actions[i], program)
            binary_sequences.append(sequence)
        binary_sequences_lst.append(binary_sequences)

    return expected_nodes(base_actions, binary_sequences_lst)


def get_top_k(k, states_lst, actions_lst, envs, program_stack, base_actions):
    # phase one - leave one out style
    programs = []
    options = []
    for i in range(len(envs)):
        program_set = []
        options_set = []
        new_states_lst = copy.deepcopy(states_lst)
        new_actions_lst = copy.deepcopy(actions_lst)
        best_value = np.inf
        for _ in range(k):
            best = np.inf
            best_i = None
            best_j = None
            best_options = None
            for j in range(len(program_stack[i])):
                value, option_lengths = one_leave_out_val(
                    new_states_lst,
                    new_actions_lst,
                    program_set + [program_stack[i][j]],
                    i,
                    base_actions,
                )
                if value < best:
                    best = value
                    best_i = i
                    best_j = j
                    best_options = option_lengths

            program_set.append(program_stack[best_i][best_j])
            options_set = best_options

            if best <= best_value:
                best_value = best
            else:
                break
        programs += program_set
        options += options_set

    # phase two - use all data
    new_states_lst = copy.deepcopy(states_lst)
    new_actions_lst = copy.deepcopy(actions_lst)
    final_programs = []
    option_sizes = []
    best_value = np.inf
    for _ in range(k):
        best = np.inf
        best_program = None
        best_options = None

        for program in programs:
            value, option_lengths = one_leave_out_val(
                new_states_lst,
                new_actions_lst,
                final_programs + [program],
                -1,
                base_actions,
            )
            if value < best:
                best = value
                best_program = program
                best_options = option_lengths

        if best < best_value:
            best_value = best
        else:
            break

        final_programs.append(best_program)
        option_sizes = best_options

    return final_programs, option_sizes
