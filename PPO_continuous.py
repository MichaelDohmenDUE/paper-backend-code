import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
import numpy as np
from backend.Utils.src.utils import gae, discounted_cumulative_reward, temporal_difference_residuals
from backend.Utils.src.ReplayBuffer import ReplayBuffer
""" 
PPO is based on the policy gradient theorem and can handle both
discrete and continuous action spaces. It can be applied to low and high dimensional observation spaces. 
In case of a a high dimensional observation space, the observation space's dimensionality is reduced by
the use of convolutional networks.
"""


# First we do low dimensional with discrete action space. So only MLP is needed


class Actor(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int, action_size: int, action_scale: int):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, hidden_size)
        self.action_mu_head = nn.Linear(hidden_size, action_size)
        self.action_log_std_head = nn.Linear(hidden_size, action_size)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_mu_head(x)
        log_std = self.action_log_std_head(x)
        clamped_log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(clamped_log_std)

        distribution = torch.distributions.normal.Normal(mu, std)
        action = distribution.rsample()
        squashed_action = F.tanh(action) * self.action_scale

        log_prob = distribution.log_prob(action).sum(axis=-1)
        #log_prob_correction = torch.log(1 - torch.pow(F.tanh(action), 2) + 1e-6)
        #log_prob = log_prob_action - log_prob_correction
        return action, log_prob


if __name__ == '__main__':
    # Data Collection & Advantage Estimation

    num_epochs = 10
    num_iterations = 500

    GAMMA = 0.99
    LAMBD = 0.95
    EPSILON = 0.2

    env = gym.make("InvertedPendulum-v5")
    observation_size = env.observation_space.shape[0]
    hidden_size = 128
    output_size = 1
    action_size = env.action_space.shape[0]
    batch_size = 64

    actor = Actor(observation_size, hidden_size, action_size, env.action_space.high)

    critic = nn.Sequential(nn.Linear(observation_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, output_size))

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=4e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=4e-4)

    actors = (actor,)
    state, _ = env.reset()

    N = len(actors)
    rollout_length = 1024

    rollout_buffer = ReplayBuffer(buffer_size=N * rollout_length)
    episode_reward = 0
    episode = 0
    for iteration in range(num_iterations):
        advantages = []
        old_action_probs = []
        total_reward = 0
        for actor in actors:
            ###################
            # Data Collection #
            ###################
            rollout = []
            rollout_old_action_probs = []
            for _ in range(rollout_length):
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
                action_tensor, action_log_probs_tensor = actor(state_tensor)
                next_state, reward, terminated, truncated, _ = env.step([action_tensor.item()])

                done = terminated or truncated

                transition = (
                    state_tensor.detach().numpy().squeeze(),
                    action_tensor.detach().numpy(),
                    reward,
                    next_state,
                    done)

                rollout.append(transition)
                rollout_old_action_probs.append(action_log_probs_tensor.detach().numpy())

                state = next_state
                episode_reward += reward

                if done:
                    state, _ = env.reset()
                    if episode % 10 == 0:
                        print(f"Episode {episode}: {episode_reward}")
                    episode_reward = 0
                    episode += 1

            ########################
            # Advantage estimation #
            ########################
            states, actions, rewards, next_states, dones = zip(*rollout)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)

            with torch.no_grad():
                state_values = critic(states_tensor).numpy()
                next_state_values = critic(next_states_tensor).numpy()

            deltas = temporal_difference_residuals(
                gamma=GAMMA,
                rewards=np.array(rewards),
                state_values=state_values,
                next_state_values=next_state_values,
                dones=np.array(dones)
            )

            rollout_advantages = gae(
                gamma=GAMMA,
                lambda_=LAMBD,
                deltas=deltas,
                dones=dones
            )

            ##################
            # Data Buffering #
            ##################
            rollout_buffer.extend(rollout)
            advantages.extend(rollout_advantages)
            old_action_probs.extend(rollout_old_action_probs)

        ####################
        # Training process #
        ####################
        old_action_probs = torch.tensor(np.array(old_action_probs), dtype=torch.float32).squeeze()
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)

        for _ in range(num_epochs):
            idx = np.random.choice(range(len(rollout_buffer) - 1), batch_size)

            batch = rollout_buffer.choice(idx)
            batch_advantages = advantages[idx]
            batch_old_action_probs = old_action_probs[idx]

            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).squeeze(-1)

            new_actions, new_action_probs = actor(states_tensor)

            log_ratio = (new_action_probs - batch_old_action_probs)
            ratio = torch.exp(log_ratio)

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
