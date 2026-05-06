import copy
import time

import numpy as np
import torch
import wandb
from torch import optim

from DeterministicPolicy.DDPG.src.Actor import Actor
from DeterministicPolicy.DDPG.src.Critic import Critic
from backend.DeterministicPolicy.TD3.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DeterministicPolicy.TD3.src.TD3TrainerProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_trainer(trainer, env_handler, eval_episodes=5):
    trainer.actor.eval()

    episode_returns = []
    current_episode_reward = 0.0

    state = env_handler.reset()
    if isinstance(state, tuple):
        state = state[0]
    while len(episode_returns) < eval_episodes:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            action = trainer.actor(state_t).cpu().numpy()

        next_state, reward, done, _ = env_handler.step(action)
        r = reward[0] if isinstance(reward, np.ndarray) else reward
        current_episode_reward += r
        if done[0]:
            episode_returns.append(current_episode_reward)
            current_episode_reward = 0.0

        state = next_state

    trainer.actor.train()

    metrics = {
        "eval/eval_avg_reward": float(np.mean(episode_returns)),
    }

    return metrics


def main(seed, env_name):
    start_time = time.time()
    env_name = env_name
    seed = seed
    max_timesteps = 1_000_000
    warmup = 25000
    eval_freq = 100000
    eval_episodes = 5
    expl_noise = 0.1
    batch_size = 256
    sync_freq = 2
    learning_rate = 3e-4
    tau = 0.005
    noise_clip = 0.5
    policy_noise = 0.2
    hidden_dim = 256
    buffer_size = int(1e6)
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="my-Td3-benchmarks",
        name=f"TD3_{env_name}_seed{seed}",
        tags=["v1.0-benchmark", "official-run"],
        config={
            "env_id": env_name,
            "seed": seed,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "max_timesteps": max_timesteps,
            "sync_freq": sync_freq,
            "tau": tau,
            "noise_clip": noise_clip,
            "policy_noise": policy_noise,
            "hidden_dim": hidden_dim,
            "warmup": warmup,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
            "expl_noise": expl_noise,
            "learning_rate": learning_rate,
        }
    )
    gym_factory = GymEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(gym_factory, seed)
    eval_env_handler = VecEnvironmentHandler(gym_factory, seed + 100)
    observation_size, action_size, max_action = env_handler.get_env_specs()
    observation_size = observation_size[0]
    if max_action is None:
        max_action = 1
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])

    transition_factory = TransitionFactory(spec)

    actor = Actor(observation_size, action_size, max_action, hidden_dim).to(device)

    actor_target = copy.deepcopy(actor).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)

    critic_1 = Critic(observation_size, action_size, hidden_dim).to(device)
    critic_2 = Critic(observation_size, action_size, hidden_dim).to(device)
    critic_target_1 = copy.deepcopy(critic_1).to(device)
    critic_target_2 = copy.deepcopy(critic_2).to(device)

    optimizer_critic_1 = optim.Adam(critic_1.parameters(), lr=learning_rate)
    optimizer_critic_2 = optim.Adam(critic_2.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(spec=spec, max_buffer_size=buffer_size, batch_size=batch_size)

    gl_counter = GlobalCounter()

    trainer = TrainProcessor(
        actor=actor,
        actor_target=actor_target,
        critic_1=critic_1,
        critic_2=critic_2,
        critic_target_1=critic_target_1,
        critic_target_2=critic_target_2,
        optimizer_critic_1=optimizer_critic_1,
        optimizer_critic_2=optimizer_critic_2,
        optimizer_actor=optimizer_actor,
        replay_buffer=replay_buffer,
        global_counter=gl_counter,
        max_action=max_action,
        learning_rate=learning_rate,
        start_timesteps=warmup,
        synchro_frequency=sync_freq,
        noise_clip=noise_clip * max_action,
        policy_noise=policy_noise * max_action,
        device=device,
    )

    datacollector = DataCollectionProcessor(actor, env_handler, transition_factory, replay_buffer, gl_counter,
                                            max_action, expl_noise, warmup, action_size, device)

    sync_process_critic_1 = SyncProcessor(critic_1, critic_target_1, tau, sync_freq, gl_counter)
    sync_process_critic_2 = SyncProcessor(critic_2, critic_target_2, tau, sync_freq, gl_counter)
    sync_process_actor = SyncProcessor(actor, actor_target, tau, sync_freq, gl_counter)

    for t in range(max_timesteps):
        metrics_ep = datacollector.run()
        metrics_train = trainer.run()
        sync_process_critic_1.run()
        sync_process_critic_2.run()
        sync_process_actor.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(t / (time.time() - start_time)),
                       "global_step": gl_counter.get()}

        if (t + 1) % eval_freq == 0:
            eval_metrics = eval_trainer(trainer, eval_env_handler, eval_episodes)
            all_metrics.update(eval_metrics)
        wandb.log(all_metrics, step=gl_counter.get())

    eval_metrics = eval_trainer(trainer, eval_env_handler, eval_episodes)
    all_metrics.update(eval_metrics)
    wandb.log(all_metrics, step=gl_counter.get())

    wandb.finish()


if __name__ == "__main__":
    seeds = [0, 1, 2]
    env_name = "Hopper-v4"
    for current_seed in seeds:
        main(current_seed, env_name)
