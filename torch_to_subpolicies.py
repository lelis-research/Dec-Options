import copy
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
import itertools


class MyNetwork(nn.Module):
    def __init__(self, mask, weight, bias, activation_pattern=None):
        super(MyNetwork, self).__init__()
        self.mask = mask
        self.weight = weight
        self.bias = bias
        self.activation_pattern = activation_pattern
        self.layers = []
        for i, current_weight in enumerate(weight):
            self.layers.append(
                nn.Linear(current_weight.shape[1], current_weight.shape[0])
            )
        with th.no_grad():
            for i, layer in enumerate(self.layers):
                layer.weight = nn.Parameter(weight[i])
                layer.bias = nn.Parameter(bias[i])
        self.layers = nn.ModuleList(self.layers)
        # the actions are either a Neural Network, "primitive",
        # or program index to look up at self.program_stack
        self.child = np.full(self.get_action_space(), "primitive", dtype=object)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            new_mask = np.array(self.mask[i])
            new_mask = new_mask[new_mask != 0]
            x[new_mask != 1] = F.relu(
                x[new_mask != 1]
            )  # deactive relu on masks with value 1
        # x = F.softmax(self.layers[-1](x))
        x = self.layers[-1](x)
        return x

    def predict(self, x, epsilon=1.0):
        deterministic = True
        if np.random.uniform() <= epsilon:
            deterministic = False
        if deterministic:
            return np.argmax(self(x).detach().numpy())
        values = F.softmax(self(x), dim=0)
        return np.random.choice(len(values), p=values.detach().numpy())

    def get_distribution(self, x):
        return F.softmax(self(x), dim=0)

    def get_action_space(self):
        return self.layers[-1].out_features

    def set_programs(self, programs):
        self.child = copy.copy(programs)

    def set_program_stack(self, program_stack):
        self.program_stack = program_stack

    def predict_hierarchical(self, x, epsilon=1.0):
        deterministic = True
        if np.random.uniform() <= epsilon:
            deterministic = False
        action = self.predict(x, epsilon)
        if self.child[action] == "primitive":
            return action
        elif isinstance(self.child[action], PPO):
            return int(self.child[action].predict(x, deterministic=deterministic)[0])
        elif isinstance(self.child[action], int):
            return self.program_stack[self.child[action]].predict_hierarchical(
                x, epsilon
            )

    def get_distribution_hierarchical(self, x, N_actions):
        probs = np.zeros(N_actions)
        dist = self.get_distribution(x).detach().numpy()
        for i, distribution in enumerate(dist):
            if self.child[i] == "primitive":
                probs[i] += distribution
            elif isinstance(self.child[i], PPO):
                d = (
                    self.child[i]
                    .policy.get_distribution(x.reshape(1, -1))
                    .distribution.probs[0]
                    .detach()
                    .numpy()
                )
                probs += d * distribution
            elif isinstance(self.child[i], int):
                d = self.program_stack[self.child[i]].get_distribution_hierarchical(
                    x, N_actions
                )
                probs += d * distribution
        return probs


def create_new_model(model, mask, activation_pattern=None):
    weight = []
    bias = []
    for i in model.children():
        if isinstance(i, th.nn.Linear):
            weight.append(i.state_dict()["weight"])
            bias.append(i.state_dict()["bias"])

    assert len(weight) == len(mask) + 1
    assert len(bias) == len(mask) + 1

    for i, current_mask in enumerate(mask):
        # remove rows
        t = np.array(current_mask)
        weight[i] = weight[i][t != 0, :]
        bias[i] = bias[i][t != 0]
        # remove column from next layer
        weight[i + 1] = weight[i + 1][:, t != 0]

    network = MyNetwork(mask, weight, bias, activation_pattern=activation_pattern)
    network.eval()
    return network


def get_dim(model):
    dims = []
    for i in model.children():
        if isinstance(i, th.nn.Linear):
            dims.append(i.in_features)
    return dims


def generate_permutations(x):
    elements = [-1, 0, 1]
    return list(itertools.product(elements, repeat=x))


def all_programs(model):
    model.policy.to("cpu")
    model_seq = copy.deepcopy(model.policy.mlp_extractor.policy_net)
    model_seq.add_module("output", model.policy.action_net)

    dim = get_dim(model_seq)
    # only one dim for now
    all_masks = generate_permutations(dim[1])

    programs = []
    # create nn for masks
    for j in all_masks:
        programs.append(create_new_model(model_seq, [j]))

    return programs


def nn_decomposition(model):
    return all_programs(model)
