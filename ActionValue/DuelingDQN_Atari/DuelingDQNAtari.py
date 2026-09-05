import os
import time
from copy import deepcopy

import torch
import wandb
from dotenv import load_dotenv

from ActionValue.DuelingDQN_Atari.src.BehaviourAtari import BehaviourAtari
from Utils.src.EnvFactory import AtariEnvFactory
from ActionValue.DQN.DQN import evaluate_policy
from ActionValue.DuelingDQN_Atari.src.EpsilonGreedy import EpsilonGreedyPolicy
from ActionValue.DuelingDQN_Atari.src.DataCollectionProcessorAtari import DataCollectionProcessor
from ActionValue.DuelingDQN_Atari.src.TrainProcessorAtari import TrainProcessor
from Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from Utils.src.EnviromentHandler import VecEnvironmentHandler
from Utils.src.ReplayBuffer import ReplayBuffer
from Utils.src.SyncProcessor import SyncProcessor
from Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, evn_name):
    start_time = time.time()
    epsilon = 1.0
    epsilon_final = 0.05
    epsilon_decay = 1_000_000
    eval_episodes = 10
    env_name = evn_name
    sync_freq = 2000
    train_freq = 4
    batch_size = 32
    max_buffer_size = 500_000
    tau = 1.0
    gamma = 0.99
    max_steps = 1_000_0000
    seed = seed
    offset = 100
    lr = 1e-4
    warmup_steps = 80_000
    setting_global_seed(seed)
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb.init(
        entity=wandb_entity,
        project="my-DuelingDQN-benchmarks",
        name=f"DuelingDQN{env_name}_seed-{seed}",
        tags=["benchmarking", "DuelingDQN"],
        config={
            "env_id": env_name,
            "exp_name": f"DuelingDQN{env_name}_seed-{seed}",
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
            "device": device,
            "epsilon_decay": epsilon_decay,
            "epsilon_final": epsilon_final,
        }
    )

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    gym_factory = AtariEnvFactory(env_name)

    env = VecEnvironmentHandler(gym_factory, seed, num_envs=1, is_eval=False)
    eval_env = VecEnvironmentHandler(gym_factory, seed + offset, num_envs=1, is_eval=True)
    obs_size, action_size, _ = env.get_env_specs()
    behavior_net = BehaviourAtari(action_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    pointer_mapping = {
        "next_state": {"source": "state", "offset": 1}
    }

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size, pointers=pointer_mapping)

    eps_greedy = EpsilonGreedyPolicy(epsilon_start=epsilon, epsilon_final=epsilon_final, epsilon_decay=epsilon_decay)
    collector = DataCollectionProcessor(behavior_net, env, buffer, eps_greedy, factory, device)

    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device, warmup_steps)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        metrics = {"global_step": step}
        metrics_data = collector.run()
        if metrics_data:
            metrics.update(metrics_data)
        if step % train_freq == 0:
            metrics_train = train_process.run()
            if metrics_train:
                metrics.update(metrics_train)
        sync_process.run()
        metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon
        metrics["global_step"] = step
        if step % 10_000 == 0:
            metrics["charts/SPS"] = int(step / max(time.time() - start_time, 1e-6))
            metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon

        if step % 10_000 == 0 and step > warmup_steps:
            metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=eval_episodes, device=device)

        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    step = max_steps
    metrics = {"global_step": step,
               "charts/eval_avg_score": evaluate_policy(behavior_net, eval_env, episodes=eval_episodes, device=device)}
    wandb.log(metrics, step=step)
    wandb.finish()
if __name__ == '__main__':
    seeds = [0,1,2]
    evn_name = "PongNoFrameskip-v4"
    for seed in seeds:
        main(seed, evn_name)
