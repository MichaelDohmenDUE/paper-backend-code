import time
from copy import deepcopy

import torch
import wandb

from backend.ActionValue.DDQN_Atari.src.DataCollectionProcessorAtari import DataCollectionProcessor
from backend.ActionValue.DDQN_Atari.src.TrainProcessorAtari import TrainProcessor
from backend.ActionValue.DQN.DQN import evaluate_policy
from backend.ActionValue.DQN.src.EpsilonGreedy import EpsilonGreedyPolicy
from backend.CommonModels.src.BehaviourAtariDQN import BehaviourAtariDQN
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import AtariEnvFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, env_name):
    start_time = time.time()
    epsilon = 1.0
    epsilon_final = 0.05
    epsilon_decay = 1_000_000
    eval_episodes = 5
    evaL_frequency = 100_000
    env_name = env_name
    sync_freq = 1000
    batch_size = 32
    max_buffer_size = 500_000
    tau = 1.0
    gamma = 0.99
    max_steps = 10_000_000
    seed = seed
    offset = 100
    lr = 1e-4
    warmup_steps = 80_000

    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="my-ddqn-benchmarks",
        name=f"DDQN_{env_name}_seed{seed}",
        tags=["benchmarking", "DDQN"],
        config={
            "env_id": env_name,
            "exp_name": f"DDQN_{env_name}_seed-{seed}",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "gamma": gamma,
            "sync_freq": sync_freq,
            "warmup_steps": warmup_steps,
            "epsilon": epsilon,
            "tau": tau,
            "eval_episodes": eval_episodes,
            "eval_frequency": evaL_frequency,
            "device": device,
            "epsilon_decay": epsilon_decay,
            "epsilon_final": epsilon_final,
        }
    )

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    gym_factory = AtariEnvFactory(env_name)

    env = VecEnvironmentHandler(gym_factory, seed, num_envs=1)
    eval_env = VecEnvironmentHandler(gym_factory, seed + offset, num_envs=1)
    obs_size, action_size, _ = env.get_env_specs()

    behavior_net = BehaviourAtariDQN(action_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    pointer_mapping = {
        "next_state": {"source": "state", "offset": 1}
    }

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size, pointers=pointer_mapping)
    eps_greedy = EpsilonGreedyPolicy(epsilon_start=epsilon, epsilon_final=0.05, epsilon_decay=epsilon_decay)
    collector = DataCollectionProcessor(behavior_net, env, buffer, eps_greedy, factory, device)

    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        metrics = {"global_step": step}
        metrics_data = collector.run()
        if metrics_data:
            metrics.update(metrics_data)
        if step % 4 == 0 and step > 0:
            metrics_train = train_process.run()
            if metrics_train:
                metrics.update(metrics_train)
        sync_process.run()
        metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon
        metrics["global_step"] = step
        if step % 10000 == 0:
            metrics["charts/SPS"] = int(step / max(time.time() - start_time, 1e-6))
            metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon

        if step % evaL_frequency == 0 and step > warmup_steps:
            metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=eval_episodes,
                                                               device=device)

        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    step = max_steps
    metrics = {"global_step": step,
               "charts/eval_avg_score": evaluate_policy(behavior_net, eval_env, episodes=eval_episodes, device=device)}
    wandb.log(metrics, step=step)
    wandb.finish()


if __name__ == '__main__':
    seeds = [0, 1, 2]
    env_name = "BreakoutNoFrameskip-v4"
    for seed in seeds:
        main(seed, env_name)
