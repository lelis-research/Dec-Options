from minigrid.wrappers import ViewSizeWrapper
from minigrid.core.world_object import Goal
import numpy as np
import gym
import gymnasium


class MiniGridWrap(gym.Env):
    def __init__(
        self,
        env,
        seed=None,
        n_discrete_actions=3,
        view_size=5,
        max_episode_steps=19 * 19,
        step_reward=0,
        show_direction=True,
    ):
        super(MiniGridWrap, self).__init__()
        # Define action and observation space
        self.seed_ = seed
        self.show_direction = show_direction
        self.step_reward = step_reward
        self.env = ViewSizeWrapper(env, agent_view_size=view_size)
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.reset()
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        shape = (len(self.observation()),)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float64
        )

        self.spec = self.env.spec
        # self.goal_position = [
        #     x for x, y in enumerate(self.env.grid.grid) if isinstance(y, Goal)
        # ]
        # self.goal_position = (
        #     int(self.goal_position[0] / self.env.height),
        #     self.goal_position[0] % self.env.width,
        # )
        # self.agent_pos = self.env.agent_pos

    def one_hot_encode(self, observation):
        OBJECT_TO_ONEHOT = {
            0: [0, 0, 0, 0],
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            8: [0, 0, 1, 0],
            10: [0, 0, 0, 1],
        }
        one_hot = [OBJECT_TO_ONEHOT[int(x)] for x in observation]
        return np.array(one_hot).flatten()

    def one_hot_encode_direction(self, direction):
        OBJECT_TO_ONEHOT = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1],
        }
        return OBJECT_TO_ONEHOT[direction]

    def observation(self):
        obs = self.env.gen_obs()
        image = self.one_hot_encode(
            self.env.observation(obs)["image"][:, :, 0].flatten()
        )
        direction = self.one_hot_encode_direction(
            self.env.observation(obs)["direction"]
        )
        if self.show_direction:
            return np.concatenate((image, direction))
        return image

    def take_basic_action(self, action):
        _, reward, terminal, _, _ = self.env.step(action)
        if terminal:
            reward = 1
        return (terminal, reward)

    def step(self, action):
        terminal, reward = self.take_basic_action(action)
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            terminal = True
        if terminal:
            self.reset()
        return (self.observation(), reward + self.step_reward, bool(terminal), {})

    def reset(self, seed=None):
        self.steps = 0
        if seed is not None:
            self.seed_ = seed
        self.env.reset(seed=self.seed_)
        return self.observation()

    def render(self):
        return self.env.render()

    def seed(self, seed):
        self.seed_ = seed
        self.env.reset(seed=seed)


def get_training_tasks_simplecross():
    env_list = []
    for i in [0, 1, 2]:
        env_list.append(
            MiniGridWrap(
                gymnasium.make("MiniGrid-SimpleCrossingS9N1-v0"),
                seed=i,
                max_episode_steps=1000,
                step_reward=-1,
            )
        )
    return env_list


def get_test_tasks_fourrooms():
    fourrooms_hard = MiniGridWrap(
        gymnasium.make("MiniGrid-FourRooms-v0"),
        seed=8,
    )
    fourrooms_medium = MiniGridWrap(
        gymnasium.make("MiniGrid-FourRooms-v0"),
        seed=51,
    )
    fourrooms_easy = MiniGridWrap(
        gymnasium.make("MiniGrid-FourRooms-v0"),
        seed=41,
    )
    return [fourrooms_easy, fourrooms_medium, fourrooms_hard]
