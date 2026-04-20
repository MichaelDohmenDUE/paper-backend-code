import time
from copy import deepcopy

import torch
import wandb

from backend.ActionValue.DQN.DQN import evaluate_policy
from backend.CommonModels.src.Policy import Policy
from backend.ActionValue.DDQN.src.TrainProcessor import TrainProcessor
from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.ActionValue.DQN.src.DataCollectionProcessor import DataCollectionProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    start_time = time.time()
    epsilon = 1.0
    env_name = "CartPole-v1"
    sync_freq = 500
    hidden_size = 64
    batch_size = 128
    max_buffer_size = 100000
    tau = 1.0
    gamma = 0.99
    max_steps = 500000
    seed = 1
    lr = 2.5e-4
    warmup_steps = 2000

    wandb.init(
        entity="michael_dohmen-",
        project="my-ddqn-benchmarks",
        config={
            "env_id": env_name,
            "exp_name": "DDQN-CartPole-v1",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "gamma": gamma,
            "sync_freq": sync_freq,
            "epsilon": epsilon,
            "tau": tau,
            "warmup_steps": warmup_steps,
        }
    )

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    gym_factory = GymEnvFactory(env_name)

    env = EnvironmentHandler(gym_factory, seed)
    eval_env = EnvironmentHandler(gym_factory, seed+1)
    obs_size, action_size, _ = env.get_env_specs()

    behavior_net = Policy(obs_size, action_size, hidden_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)
    eps_greedy = EpsilonGreedyPolicy(epsilon_start=epsilon, epsilon_final=0.05, epsilon_decay=20000)
    collector = DataCollectionProcessor(behavior_net, env, buffer, eps_greedy, factory, device)

    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        metrics = None
        collector.run()
        metrics = train_process.run()
        sync_process.run()

        if metrics and step % 400 == 0:
            metrics["charts/SPS"] = int(step / (time.time() - start_time))
            metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon
            wandb.log(metrics, step=step)
        if step % 10_000 == 0 and step > warmup_steps:
            avg_score = evaluate_policy(behavior_net, eval_env, episodes=10, device=device)
            wandb.log({"charts/eval_avg_score": avg_score}, step=step)

if __name__ == "__main__":
    main()
