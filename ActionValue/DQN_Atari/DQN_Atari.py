import os
import time
from copy import deepcopy

import torch
import wandb
from dotenv import load_dotenv

from ActionValue.DQN_Atari.src.EpsilonGreedy import EpsilonGreedyPolicy
from ActionValue.DQN_Atari.src.DataCollectionProcessorAtari import DataCollectionProcessor
from ActionValue.DQN_Atari.src.TrainProcessorAtari import TrainProcessor
from ActionValue.DQN_Atari.src import BehaviourAtariDQN
from ActionValue.DQN_Atari.src.BehaviourAtariDQN import BehaviourAtariDQN
from Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from Utils.src.EnvFactory import AtariEnvFactory
from Utils.src.EnviromentHandler import VecEnvironmentHandler
from Utils.src.ReplayBuffer import ReplayBuffer
from Utils.src.SyncProcessor import SyncProcessor
from Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

def evaluate_policy(policy, env_handler, episodes=5, device="cpu"):
    policy.eval()
    total_reward = 0.0

    for _ in range(episodes):
        state = env_handler.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, device=device).float()
            with torch.no_grad():
                q_values = policy(state_tensor)
                action_int = torch.argmax(q_values, dim=1).item()

            next_state, reward, done_batch, _ = env_handler.step([action_int])

            episode_reward += reward[0]
            state = next_state
            done = done_batch[0]

        total_reward += episode_reward

    policy.train()
    return total_reward / episodes


def main(seed, env_name):
    # initialization
    start_time = time.time()
    lr = 1e-4
    env_name = env_name
    sync_freq = 10_000
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = 1_000_000
    batch_size = 32
    max_buffer_size = 500_000
    tau = 1.0
    gamma = 0.99
    max_steps = 10_000_000
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    gym_factory = AtariEnvFactory(env_name)
    factory = TransitionFactory(spec)
    max_norm = 5
    seed = seed
    warmup_steps = 50000
    setting_global_seed(seed)
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb.init(
        entity=wandb_entity,
        project="my-dqn-benchmarks",
        name=f"DQN_{env_name}_seed{seed}",
        tags=["testing", "DQN"],
        config={
            "env_id": env_name,
            "exp_name": f"DQN_{env_name}_seed-{seed}",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "gamma": gamma,
            "sync_freq": sync_freq,
            "warmup_steps": warmup_steps,
            "max_norm": max_norm,
            "epsilon": epsilon,
            "epsilon_final": epsilon_final,
            "epsilon_decay": epsilon_decay,
            "tau": tau,
        }
    )

    env = VecEnvironmentHandler(gym_factory, seed, num_envs=1, is_eval=False)
    eval_env = VecEnvironmentHandler(gym_factory, seed + 1, num_envs=1, is_eval=True)
    obs_size, action_size, max_action = env.get_env_specs()

    behavior_net = BehaviourAtariDQN(action_size).to(device)
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr=lr)

    target_net = deepcopy(behavior_net).to(device)

    pointer_mapping = {
        "next_state": {"source": "state", "offset": 1}
    }

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size, pointers=pointer_mapping)

    eps_greedy = EpsilonGreedyPolicy(epsilon_start=epsilon, epsilon_final=epsilon_final, epsilon_decay=epsilon_decay)
    collector = DataCollectionProcessor(behavior_net, env, buffer, eps_greedy, factory, device)
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, max_norm, warmup_steps, device)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        metrics = {"global_step": step}
        metrics_data = collector.run()
        if metrics_data:
            metrics.update(metrics_data)

        if step % 4 == 0 and step >= warmup_steps:
            metrics_train = train_process.run()
            if metrics_train:
                metrics.update(metrics_train)
        sync_process.run()

        if step % 400 == 0:
            metrics["charts/SPS"] = int(step / max(time.time() - start_time, 1e-6))
            metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon

        if step % 10_000 == 0 and step > warmup_steps:
            metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=5, device=device)

        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=5, device=device)
    wandb.log(metrics, step=step)
    wandb.finish()

if __name__ == '__main__':
    seeds = [0, 1, 2]
    for seed in seeds:
        main(seed, env_name="BreakoutNoFrameskip-v4")
