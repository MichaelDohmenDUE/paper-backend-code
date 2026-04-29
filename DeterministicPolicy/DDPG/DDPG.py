import time
from copy import deepcopy

import numpy as np
import torch
import wandb

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.DeterministicPolicy.DDPG.src.ActionHandler import OUNoise, DeterministicPolicyWithNoise
from backend.DeterministicPolicy.DDPG.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DeterministicPolicy.DDPG.src.TrainProcessor import TrainProcess
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    trainer.actor.eval()

    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            with torch.no_grad():
                action = trainer.actor(state_t).cpu().numpy().flatten()
            next_state, reward, terminated, truncated, _ = env_handler.step_ddpg(action)
            avg_reward += reward
            state = next_state
            done = terminated or truncated
    avg_reward /= eval_episodes
    metrics = {
        "eval/eval_avg_reward": avg_reward,
    }
    trainer.actor.train()
    return metrics

def main(seed, env_name):
    start_time = time.time()
    lr_actor = 1e-4
    lr_critic = 1e-3
    max_timesteps = 1_000_000
    env_name = env_name
    sync_freq = 1
    hidden_size = 300
    batch_size = 256
    max_buffer_size = 1000000
    tau = 0.001
    gamma = 0.99
    seed = seed
    warmup = 25000
    eval_freq = 5000
    eval_episodes = 10
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="my-DDPG-benchmarks",
        name=f"DDPG_{env_name}_seed_{seed}",
        tags=["v1.0-benchmark", "official-run"],
        config={
            "env_id": env_name,
            "exp_name": f"DDPG-{env_name}",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "max_timesteps": max_timesteps,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "gamma": gamma,
            "sync_freq": sync_freq,
            "tau": tau,
        }
    )

    gym_factory = GymEnvFactory(env_name)
    env = EnvironmentHandler(gym_factory, seed)
    eval_env_handler = EnvironmentHandler(gym_factory, seed + 100)
    observation_size, action_size, max_action = env.get_env_specs()

    # Networks
    actor = Actor(observation_size, action_size, max_action, hidden_size).to(device)
    actor_target = deepcopy(actor).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

    critic = Critic(observation_size, action_size, hidden_size).to(device)
    critic_target = deepcopy(critic).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=0.01)

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    noise = OUNoise(action_size, theta=0.15, sigma=0.2, device=device)
    #policy = DeterministicPolicyWithNoise(actor, noise, max_action, device)

    gl_counter = GlobalCounter()

    data_collection_process = DataCollectionProcessor(env, actor, noise, buffer, factory, gl_counter,max_action, device)
    train_process = TrainProcess(buffer, actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer,
                                 gamma, warmup, device)

    sync_process_actor = SyncProcessor(actor, actor_target, tau, sync_freq, gl_counter)
    sync_process_critic = SyncProcessor(critic, critic_target, tau, sync_freq, gl_counter)

    for t in range(max_timesteps):
        metrics_ep = data_collection_process.run()
        metrics_train = train_process.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(t / (time.time() - start_time)),
                       "global_step": gl_counter.get()}
        if (t + 1) % eval_freq == 0:
            eval_metrics = eval_trainer(train_process, eval_env_handler, eval_episodes)
            all_metrics.update(eval_metrics)
        wandb.log(all_metrics, step=gl_counter.get())
        sync_process_actor.run()
        sync_process_critic.run()
    wandb.finish()
if __name__ == "__main__":
    seeds = [1, 2, 3]
    env_name = "HalfCheetah-v4"
    for seed in seeds:
        main(seed, env_name)
