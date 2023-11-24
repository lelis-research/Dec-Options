from environment_combo4 import get_training_tasks_combo4, get_test_task_combo4
from environment_minigrid import get_training_tasks_simplecross
from environment_minigrid import get_test_tasks_fourrooms
import argparse
from baselines import training_tasks_learning
import json
from test_tasks import test_tasks


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
    parser.add_argument("--log_path", default="logs/")
    parser.add_argument(
        "--baseline",
        default="Vanilla",
        choices=["Vanilla", "NeuralAugmented", "DecOptionsWhole", "DecOptions"],
    )
    parser.add_argument("--parameter_sweep", default=False, type=bool)
    args_p = parser.parse_args()

    with open(args_p.config) as f:
        c = json.load(f)
        training_args = c["TrainingTasks"]
        config = c[str(args_p.phase)]
        hyperparameter_seach_space = c["search_space"]
        training_args["log_path"] = args_p.log_path
        config["log_path"] = args_p.log_path

    if config["task"] == "fourrooms":
        training_tasks = get_training_tasks_simplecross()
        env = get_test_tasks_fourrooms()[config["difficulty"]]
    elif config["task"] == "combogrid":
        training_tasks = get_training_tasks_combo4(config["size"])
        env = get_test_task_combo4([config["size"]])

    if args_p.phase == "TrainingTasks":
        training_tasks_learning(training_tasks, args_p.seed, training_args)
        return

    if args_p.phase == "TestTasks":
        test_tasks(
            args_p,
            hyperparameter_seach_space,
            config,
            len(training_tasks) + 1,
            env,
            training_tasks,
        )


if __name__ == "__main__":
    main()
