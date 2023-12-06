import gym
import numpy as np

MOVE_PATTERN = [[0, 2, 2, 1], [0, 0, 1, 1], [1, 2, 1, 0], [1, 0, 2, 2]]
COMBO_SIZE = 4
NUM_ACTIONS = 3


class ComboGridWorld(gym.Env):
    def __init__(
        self,
        size,
        max_episode_steps=500,
        terminal_f=None,
        state_f=None,
        n_discrete_actions=3,
    ):
        """ """
        super(ComboGridWorld, self).__init__()
        self.seed_ = None
        self.n_discrete_actions = n_discrete_actions
        self.max_episode_steps = max_episode_steps
        self.size = size
        self.terminal_f = terminal_f
        self.state_f = state_f
        self.patterns = MOVE_PATTERN

        self.last_actions = None
        self.last_actions_count = None
        self.state = None
        self.terminal = None
        self.steps = 0

        self.grid = np.ones((size, size), dtype=bool)

        self.set_walls()
        self.set_terminal()

        self.reset()
        self.action_space = gym.spaces.Discrete(self.n_discrete_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(self.observation()),), dtype=np.float64
        )

    def seed(self, seed):
        self.seed_ = seed
        self.reset()

    def set_walls(self):
        # wall around the grid
        for i in range(self.size):
            self.grid[i][0] = 0
            self.grid[0][i] = 0
            self.grid[i][self.size - 1] = 0
            self.grid[self.size - 1][i] = 0
        # wall in the middle of the grid
        for i in range(2, self.size - 2):
            for j in range(1, int((self.size - 1) / 2)):
                self.grid[i][j] = 0
                self.grid[i][j + int(self.size / 2)] = 0

    def set_terminal(self):
        if self.terminal_f is not None:
            self.terminal_f(self)
            return

    def normalize_o(self):
        # remove the surronding walls from the observation
        one_hot_encode = np.zeros((self.size - 2, self.size - 2))
        one_hot_encode[self.state[0] - 1][self.state[1] - 1] = 1
        one_hot_encode_terminal = np.zeros((self.size - 2, self.size - 2))
        one_hot_encode_terminal[self.terminal[0] - 1][self.terminal[1] - 1] = 1
        return np.concatenate((one_hot_encode, one_hot_encode_terminal)).flatten()

    def normalize_a(self, actions):
        one_hot_encode = []
        for action in actions:
            action = int(action)
            x = np.zeros(self.n_discrete_actions)
            if action != -1:
                x[action] = 1
            one_hot_encode.append(x)
        return np.array(one_hot_encode).flatten()

    def observation(self):
        return np.concatenate(
            (self.normalize_a(self.last_actions[:-1]), self.normalize_o()),
            axis=0,
        ).astype(np.float64)

    def take_basic_action(self, action):
        next_state = np.copy(self.state)
        self.last_actions[self.last_actions_count] = action
        self.last_actions_count += 1
        if self.last_actions_count == len(self.last_actions):
            self.last_actions_count = 0
            if np.all(self.last_actions == self.patterns[0]):
                next_state += np.array([1, 0])
            elif np.all(self.last_actions == self.patterns[1]):
                next_state -= np.array([1, 0])
            elif np.all(self.last_actions == self.patterns[2]):
                next_state += np.array([0, 1])
            elif np.all(self.last_actions == self.patterns[3]):
                next_state -= np.array([0, 1])
            self.last_actions = np.zeros(len(self.last_actions)) - 1

        if self.grid[next_state[0], next_state[1]] == 1:
            self.state = np.copy(next_state)

        terminal = (self.state == self.terminal).all()
        reward = 0
        if terminal:
            reward = 1
        return (terminal, reward - 1)

    def step(self, action):
        terminal = None
        reward = 0
        terminal, reward = self.take_basic_action(action)
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            terminal = True
        if terminal:
            self.reset()
        return (self.observation(), reward, bool(terminal), {})

    def reset(self):
        if self.state_f is not None:
            self.state_f(self)
        else:
            self.state = np.array(
                [
                    np.random.randint(1, self.size - 1),
                    np.random.randint(1, self.size - 1),
                ]
            )
        self.last_actions = np.zeros(len(self.patterns[0])) - 1
        self.last_actions_count = 0
        self.steps = 0
        return self.observation()

    def __str__(self):
        _str = ""
        for i in range(self.size):
            _str += "____" * self.size + "\n"
            _str += "|"
            for j in range(self.size):
                if (np.array([i, j]) == self.terminal).all():
                    _str += "*" + " | "
                elif (np.array([i, j]) == self.state).all():
                    _str += "$" + " | "
                elif self.grid[i, j] == 0:
                    _str += "#" + " | "
                else:
                    _str += " " + " | "
            _str += "\n"
        _str += "____" * self.size + "\n"
        return _str

    def render(self):
        grid = np.zeros(self.grid.shape)  # floor
        grid[self.grid == False] = 1  # wall
        for i in self.terminals:
            grid[i[0], i[1]] = 2  # terminal
        grid[self.state[0], self.state[1]] = 3  # agent
        # Calculate the size of the grid image
        cell_size = 30  # Size of each cell in pixels
        grid_height, grid_width = grid.shape
        image_height = grid_height * cell_size
        image_width = grid_width * cell_size

        # Create a blank image
        grid_image = (
            np.ones((image_height, image_width, 3), dtype=np.uint8) * 192
        )  # Gray background
        colors = {1: [192, 192, 192], 0: [0, 0, 0], 3: [255, 0, 0], 2: [0, 255, 0]}

        for i in range(grid_height):
            for j in range(grid_width):
                cell = grid[i, j]
                cell_color = colors.get(cell, "black")
                # Fill the cell with the appropriate color
                grid_image[
                    i * cell_size + 1 : (i + 1) * cell_size - 1,
                    j * cell_size + 1 : (j + 1) * cell_size - 1,
                ] = cell_color

        return grid_image


class TestTask(ComboGridWorld):
    def __init__(self, size, max_episode_steps=500, n_discrete_actions=3):
        self.terminals = None
        super(TestTask, self).__init__(
            size, max_episode_steps, n_discrete_actions=n_discrete_actions
        )

    def set_walls(self):
        for i in range(self.size):
            self.grid[i][0] = 0
            self.grid[0][i] = 0
            self.grid[i][self.size - 1] = 0
            self.grid[self.size - 1][i] = 0

    def set_terminal(self):
        center1 = int(self.size / 2)
        center2 = int((self.size - 1) / 2)
        self.terminals = np.array(
            [
                [1, center1],
                [center1, 1],
                [center1, self.size - 2],
                [self.size - 2, center2],
            ]
        )

    def normalize_o(self):
        one_hot_encode = np.zeros((self.size - 2, self.size - 2))
        one_hot_encode[self.state[0] - 1][self.state[1] - 1] = 1
        one_hot_encode_terminal = np.zeros((self.size - 2, self.size - 2))
        for i in self.terminals:
            one_hot_encode_terminal[i[0] - 1][i[1] - 1] = 1
        return np.concatenate((one_hot_encode, one_hot_encode_terminal)).flatten()

    def take_basic_action(self, action):
        next_state = np.copy(self.state)
        self.last_actions[self.last_actions_count] = action
        self.last_actions_count += 1
        if self.last_actions_count == len(self.last_actions):
            self.last_actions_count = 0
            if np.all(self.last_actions == self.patterns[0]):
                next_state += np.array([1, 0])
            elif np.all(self.last_actions == self.patterns[1]):
                next_state -= np.array([1, 0])
            elif np.all(self.last_actions == self.patterns[2]):
                next_state += np.array([0, 1])
            elif np.all(self.last_actions == self.patterns[3]):
                next_state -= np.array([0, 1])
            self.last_actions = np.zeros(len(self.last_actions)) - 1

        if self.grid[next_state[0], next_state[1]] == 1:
            self.state = np.copy(next_state)

        reward = 0
        is_terminal = False

        for i, terminal in enumerate(self.terminals):
            if (self.state == terminal).all():
                reward = 10
                self.terminals = np.delete(self.terminals, i, 0)
                break

        if len(self.terminals) == 0:
            is_terminal = True

        return (is_terminal, reward)

    def reset(self):
        self.set_terminal()
        self.state = np.array([int(self.size / 2), int(self.size / 2)])
        self.last_actions = np.zeros(len(self.patterns[0])) - 1
        self.last_actions_count = 0
        self.steps = 0
        return self.observation()


def terminal1(obj):
    obj.terminal = np.array([1, 1])
    if obj.grid[obj.terminal[0], obj.terminal[1]] == 0:
        obj.set_terminal()


def state1(obj):
    obj.state = np.array([obj.size - 2, 1])


def terminal2(obj):
    obj.terminal = np.array([1, obj.size - 2])
    if obj.grid[obj.terminal[0], obj.terminal[1]] == 0:
        obj.set_terminal()


def state2(obj):
    obj.state = np.array([obj.size - 2, obj.size - 2])


def terminal3(obj):
    obj.terminal = np.array([obj.size - 2, 1])
    if obj.grid[obj.terminal[0], obj.terminal[1]] == 0:
        obj.set_terminal()


def state3(obj):
    obj.state = np.array([1, 1])


def terminal4(obj):
    obj.terminal = np.array([obj.size - 2, obj.size - 2])
    if obj.grid[obj.terminal[0], obj.terminal[1]] == 0:
        obj.set_terminal()


def state4(obj):
    obj.state = np.array([1, obj.size - 2])


def get_training_tasks_combo4(size):
    max_steps = size * size * COMBO_SIZE * 20
    terminal_functions = [terminal1, terminal2, terminal3, terminal4]
    state_functions = [state1, state2, state3, state4]
    envs = []
    for i in range(4):
        envs.append(
            ComboGridWorld(
                size + 2,
                max_steps,
                terminal_f=terminal_functions[i],
                state_f=state_functions[i],
            )
        )

    return envs


def get_test_task_combo4(size):
    max_steps = size * size * COMBO_SIZE * 4

    return TestTask(
        size + 2,
        max_steps,
    )
