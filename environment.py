import gym
import copy
import torch as th


class basic_actions:
    def __init__(self, action):
        self.action = action

    def predict(self, x):
        return self.action

    def predict_hierarchical(self, x, epsilon):
        return self.predict(x)


class MasterEnv(gym.Env):
    def __init__(self, env, option_sizes, options, deterministic=True):
        # basic actions are options with size 1
        self.env = env
        self.options = options
        self.action_space = gym.spaces.Discrete(len(option_sizes))
        self.observation_space = env.observation_space
        self.deterministic = deterministic
        self.spec = self.env.spec
        self.option_sizes = option_sizes
        self.seed_ = None

    def __deepcopy__(self, memo):
        # Create a new instance of the class with shallow copies of all attributes
        new_obj = self.__class__(
            self.env, self.option_sizes, self.options, self.deterministic
        )
        # Add the new object to the memo dictionary
        memo[id(self)] = new_obj
        # Shallow copy the 'shallow_copy_me' attribute
        new_obj.options = self.options
        # Deepcopy all other attributes
        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "options":
                setattr(new_obj, attr_name, copy.deepcopy(attr_value, memo))
        # Return the new object
        return new_obj

    def get_basic_action(self, action, observation=None):
        if self.deterministic is True:
            epsilon = 0.0
        else:
            epsilon = 1.0
        if observation is None:
            x = th.tensor(self.observation()).float()
        else:
            x = th.tensor(observation).float()

        return self.options[action].predict_hierarchical(x, epsilon=epsilon)

    def observation(self):
        return self.env.observation()

    def step(self, action):
        option = action
        rewards = 0
        for _ in range(self.option_sizes[option]):
            action = self.get_basic_action(option)
            o, r, done, _ = self.env.step(action)
            rewards += r
            if done:
                break
        return o, rewards, done, _

    def reset(self, seed=None):
        return self.env.reset()

    def seed(self, seed):
        self.seed_ = seed
        self.reset()
