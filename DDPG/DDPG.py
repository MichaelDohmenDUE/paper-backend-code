from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory


class DataCollectionProcessor:
    def __init__(
        self,
        env: EnvironmentHandler,
        actor: nn.Module,
        buffer: ReplayBuffer,
        transition_factory: TransitionFactory,
        device: torch.device,
        noise_std: float,
        noise_mean: float = 0.0,
    ):
        self.env = env
        self.actor = actor.to(device)
        self.buffer = buffer
        self.transition_factory = transition_factory
        self.device = device
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.episode_timesteps = 0

        self.state = self.env.reset()
        self.done = False

    def run(self):
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)

        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=action.shape,
            device=self.device,
        )

        action_noisy = (action + noise).clamp(-self.env.max_action, self.env.max_action).cpu().numpy()
        self.episode_timesteps += 1
        next_state, reward, done, done_bool = self.env.step(action_noisy, episode_timesteps=self.episode_timesteps)
        transition = self.transition_factory.create(
            state=self.state,
            action=action_noisy,
            reward=reward,
            next_state=next_state,
            done=done_bool,
        )
        self.buffer.append(transition)

        self.state = next_state
        self.done = done_bool

        if self.done:
            self.state = self.env.reset()
            self.done = False
            self.episode_timesteps = 0
        return transition

class TrainProcess:
    def __init__(self, buffer: ReplayBuffer, actor: nn.Module, actor_target: nn.Module, critic: nn.Module,
                 critic_target: nn.Module, actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer, gamma, device):
        self.buffer = buffer
        self.actor = actor.to(device)
        self.actor_target = actor_target.to(device)
        self.critic = critic.to(device)
        self.critic_target = critic_target.to(device)
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.gamma = gamma
        self.device = device

    def run(self):
        if len(self.buffer) < self.buffer.batch_size:
            return None, None

        batch = self.buffer.sample_batch()

        states      = batch["state"].to(self.device)
        actions     = batch["action"].to(self.device)
        rewards     = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones       = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.gamma * target_q * (1.0 - dones)
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        # Actor update
        self.actor_opt.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss, critic_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    lr = 1e-3
    num_episodes = 1000
    env_name = "InvertedPendulum-v5"
    sync_freq = 1
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 0.001
    gamma = 0.99
    seed = 42
    noise_mean = 0.0
    noise_std = 0.1

    env = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env.get_env_specs()

    # Networks
    actor = Actor(observation_size, action_size, max_action, hidden_size).to(device)
    actor_target = deepcopy(actor).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    critic = Critic(observation_size, action_size, hidden_size).to(device)
    critic_target = deepcopy(critic).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    data_collection_process = DataCollectionProcessor(env,actor, buffer, factory, device, noise_std=noise_std, noise_mean=noise_mean)

    train_process = TrainProcess(buffer,actor,actor_target,critic,critic_target,actor_optimizer,critic_optimizer, gamma , device)
    sync_process_actor = SyncProcessor(actor, actor_target, tau, sync_freq)
    sync_process_critic = SyncProcessor(critic, critic_target, tau, sync_freq)
    total_steps = 0

    for episode in range(num_episodes):
        done = False
        episode_reward = 0.0
        actor_loss, critic_loss = None, None
        while not done:
            # Data Collection
            transition = data_collection_process.run()
            total_steps += 1
            actor_loss, critic_loss = train_process.run()
            done = transition.done
            episode_reward += transition.reward

            sync_process_actor.run()
            sync_process_critic.run()

        if episode % 10 == 9 and actor_loss is not None and critic_loss is not None:
            print(
                f"Episode: {episode + 1}, "
                f"Reward: {episode_reward:.2f}, "
                f"actor_loss: {actor_loss.mean():.3f}, "
                f"critic_loss: {critic_loss:.3f}"
            )
if __name__ == "__main__":
    main()
