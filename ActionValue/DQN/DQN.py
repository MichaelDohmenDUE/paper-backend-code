import time
from copy import deepcopy

import torch
import wandb
from torch import nn

from backend.ActionValue.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.ActionValue.DQN.src.DataCollectionProcessor import DataCollectionProcessor
from backend.ActionValue.DQN.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler, VecEnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(policy, env_handler, episodes=5, device="cpu"):
    """
    Generated with GEemini
    """
    policy.eval()
    total_reward = 0.0

    for _ in range(episodes):
        state = env_handler.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            state_tensor = torch.tensor(state, device=device).float()
            with torch.no_grad():
                q_values = policy(state_tensor)
                action_int = torch.argmax(q_values, dim=1).item()
            action_batch = [action_int]
            next_state, reward, done, _ = env_handler.step(action_batch)
            episode_reward += reward[0]
            state = next_state
            steps += 1
            done = done[0]

        total_reward += episode_reward

    policy.train()
    return total_reward / episodes

def main(seed):
    # initialization
    start_time = time.time()
    lr = 2.5e-4
    epsilon = 1.0
    epsilon_final = 0.05
    epsilon_decay = 20000
    eval_episodes = 10
    env_name = "CartPole-v1"
    sync_freq = 1000
    hidden_size = 64
    batch_size = 128
    max_buffer_size = 100_000
    tau = 1.0
    gamma = 0.99
    max_steps = 1000000
    max_norm = 5
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    gym_factory = GymEnvFactory(env_name)
    factory = TransitionFactory(spec)
    seed = seed
    warmup_steps = 500

    wandb.init(
        entity="michael_dohmen-",
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
            "tau": tau,
            "eval_episodes": eval_episodes,
            "device": device,
            "epsilon_decay": epsilon_decay,
            "epsilon_final": epsilon_final,
        }
    )

    env = VecEnvironmentHandler(gym_factory, seed, num_envs=1)
    eval_env = VecEnvironmentHandler(gym_factory, seed + 100, num_envs=1)
    obs_size, action_size, max_action = env.get_env_specs()
    obs_size = obs_size[0]
    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(), nn.Linear(hidden_size, action_size)).to(device)
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net).to(device)

    eps_greedy = EpsilonGreedyPolicy(epsilon_start=epsilon, epsilon_final=epsilon_final, epsilon_decay=epsilon_decay)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)
    collector = DataCollectionProcessor(behavior_net, env, buffer, eps_greedy, factory, device)
    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, max_norm, warmup_steps, device)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)
    for step in range(max_steps):
        metrics = {"global_step": step}
        metrics_data = collector.run()
        if metrics_data:
            metrics.update(metrics_data)
        metrics_train = train_process.run()
        if isinstance(metrics_train, dict) and metrics_train:
            metrics.update(metrics_train)
        sync_process.run()

        if step % 400 == 0:
            metrics["charts/SPS"] = int(step / max(time.time() - start_time, 1e-6))
            metrics["charts/epsilon"] = collector.epsilon_greedy.epsilon

        if step % 10_000 == 0 and step > warmup_steps:
            metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=eval_episodes, device=device)

        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    metrics = {"global_step": step}
    metrics["charts/eval_avg_score"] = evaluate_policy(behavior_net, eval_env, episodes=eval_episodes, device=device)
    wandb.log(metrics, step=step)
    wandb.finish()
if __name__ == '__main__':
    seeds = [0,1,2]
    for seed in seeds:
        main(seed)
