import torch
from torch import nn, Tensor
from torch.nn import functional as F
from collections import deque
from typing import Any, Iterable, Callable
import random
import gymnasium as gym
import numpy as np
from utils import ReplayBuffer, gae, discounted_cumulative_reward, temporal_difference_residuals

""" 
PPO is based on the policy gradient theorem and can handle both
discrete and continuous action spaces. It can be applied to low and high dimensional observation spaces. 
In case of a a high dimensional observation space, the observation space's dimensionality is reduced by
the use of convolutional networks.
"""

# First we do low dimensional with discrete action space. So only MLP is needed


if __name__ == '__main__':
    # Data Collection & Advantage Estimation

    num_epochs = 10
    num_iterations = 500

    GAMMA = 0.9
    LAMBD = 1.0
    EPSILON = 0.2

    env = gym.make("CartPole-v1")
    observation_size = env.observation_space.shape[0]
    hidden_size = 16
    output_size = 1
    action_size = env.action_space.n
    batch_size = 1000

    actor = nn.Sequential(nn.Linear(observation_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, action_size),
                          nn.Softmax(dim=1))

    critic = nn.Sequential(nn.Linear(observation_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, output_size))

    actor_optimizer = torch.optim.Adam(actor.parameters())
    critic_optimizer = torch.optim.Adam(critic.parameters())

    actors = (actor,)
    state, _ = env.reset()

    N = len(actors)
    T = 1000

    rollout_buffer = ReplayBuffer(buffer_size=N * T)
    episode_reward = 0
    episode = 0
    for iteration in range(num_iterations):
        advantages = []
        old_action_probs = []
        total_reward = 0
        for actor in actors:
            # Rollout each actor
            rollout = []
            rollout_old_action_probs = []
            for _ in range(T):
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
                action_probs_tensor = actor(state_tensor)
                action_tensor = torch.multinomial(action_probs_tensor, 1)

                next_state, reward, terminated, truncated, _ = env.step(action_tensor.item())

                done = terminated or truncated

                transition = (
                    state_tensor.detach().numpy().squeeze(),
                    action_tensor.detach().numpy(),
                    reward,
                    next_state,
                    done)

                rollout.append(transition)
                rollout_old_action_probs.append(action_probs_tensor.detach().numpy())

                state = next_state
                episode_reward += reward

                if done:
                    state, _ = env.reset()
                    if episode % 10 == 0:
                        print(f"Episode {episode}: {episode_reward}")
                    episode_reward = 0
                    episode += 1

            # Advantage estimation

            states, actions, rewards, next_states, dones = zip(*rollout)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
            dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

            with torch.no_grad():
                state_values = critic(states_tensor)
                next_state_values = critic(next_states_tensor)

            deltas = temporal_difference_residuals(
                gamma=GAMMA,
                rewards=rewards_tensor,
                state_values=state_values,
                next_state_values=next_state_values,
                dones=dones_tensor
            ).detach().numpy()

            rollout_advantages = gae(
                gamma=GAMMA,
                lambda_=LAMBD,
                deltas=deltas,
                dones=dones
            )

            rollout_buffer.extend(rollout)
            advantages.extend(rollout_advantages)
            old_action_probs.extend(rollout_old_action_probs)

        # Training process

        old_action_probs = torch.tensor(np.array(old_action_probs), dtype=torch.float32).squeeze()
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)

        for _ in range(num_epochs):
            idx = np.random.choice(range(len(rollout_buffer) - 1), batch_size)
            batch = rollout_buffer.choice(idx)
            batch_advantages = advantages[idx]

            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).squeeze(-1)

            new_action_probs = actor(states_tensor)

            batch_old_action_probs = old_action_probs[idx]

            ratio = (new_action_probs.gather(dim=1, index=actions_tensor)
                     / batch_old_action_probs.gather(dim=1,
                                                     index=actions_tensor))

            clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

            actor_optimizer.zero_grad()
            actor_loss = - batch_advantages * torch.min(clipped_ratio, ratio)
            actor_loss.mean().backward()
            actor_optimizer.step()

            rewards_to_go = torch.tensor(discounted_cumulative_reward(
                gamma=GAMMA,
                rewards=rewards,
                dones=dones
            ), dtype=torch.float32).unsqueeze(-1)
            state_values = critic(states_tensor)

            critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(state_values, rewards_to_go)
            critic_loss.backward()
            critic_optimizer.step()
