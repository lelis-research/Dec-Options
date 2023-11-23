from environment_combo4 import get_training_tasks_combo4, get_test_task_combo4
from environment_minigrid import get_training_tasks_simplecross
from environment_minigrid import get_test_tasks_fourrooms
import argparse
from baselines import training_tasks_learning
import json


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
    parser.add_argument("--baseline", default="baseline")
    args_p = parser.parse_args()

    with open(args_p.config) as f:
        training_args = json.load(f)["TrainingTasks"]
        config = json.load(f)[args_p.phase]

    if config["task"] == "fourrooms":
        training_task = get_training_tasks_simplecross()
        envs = get_test_tasks_fourrooms()[config["difficulty"]]
    elif config["task"] == "combogrid":
        training_task = get_training_tasks_combo4(config["size"])
        envs = get_test_task_combo4([config["size"]])

    if args_p.phase == "TrainingTasks":
        training_tasks_learning(training_task, args_p.seed, training_args)


if __name__ == "__main__":
    main()
