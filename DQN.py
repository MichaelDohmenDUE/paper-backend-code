import torch
from torch import nn
from collections import deque
from typing import Any, Iterable, Callable
import random
import gymnasium as gym
import numpy as np


def epsilon_greedy(q_values: torch.tensor, epsilon: float) -> torch.tensor:
    """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
        Randomly samples from max actions if there is a tie.
     """
    actions = torch.arange(len(q_values))
    max_q_value = torch.max(q_values)
    max_idx = (q_values == max_q_value).to(torch.int64)
    greedy_action = random.choice(actions[max_idx == 1])

    return random.choice(actions) if random.random() < epsilon else greedy_action


class ReplayBuffer:
    def __init__(self, buffer_size: int = 10_000):
        self.buffer: deque[Any] = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def __str__(self) -> str:
        return f"{list(self.buffer)}"

    def append(self, x: Any) -> None:
        self.buffer.append(x)

    def extend(self, iterable: Iterable[Any]) -> None:
        self.buffer.extend(iterable)

    def sample(self, batch_size: int) -> list[Any]:
        return random.sample(self.buffer, batch_size)

if __name__ == '__main__':
    # initialization
    env = gym.make("CartPole-v1")
    observation_size = env.observation_space.shape[0]
    hidden_size = 32
    action_size = env.action_space.n

    behavior_policy = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))
    target_policy = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size))

    target_policy.load_state_dict(behavior_policy.state_dict())

    optimizer = torch.optim.Adam(behavior_policy.parameters())
    criterion = nn.MSELoss()
    buffer = ReplayBuffer()

    gamma = 0.99
    num_episodes = 400
    update_freq = 40

    # Outer training loop
    total_steps = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        episode_rewards = 0
        while not done:

            # Data collection
            q_values = behavior_policy(torch.tensor(state, dtype=torch.float32))
            action = epsilon_greedy(q_values=q_values, epsilon=0.2)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            transition = (state, action.item(), reward, next_state, done)
            buffer.append(transition)
            state = next_state

            # Learning
            batch_size = 64
            if len(buffer) > batch_size:
                transitions = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*transitions)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32)  # batch_size x 4
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)  # batch_size x 1
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

                qsa_behavior = behavior_policy(states_tensor).gather(1, actions_tensor)  # ^y
                qs_target = target_policy(next_states_tensor)  # batch_size x action_dim
                qsa_target = torch.max(qs_target, dim=1).values.unsqueeze(-1).detach()
                target = rewards_tensor + gamma * qsa_target * (1.0 - dones_tensor) # [~dones_tensor]  # y
                optimizer.zero_grad()
                loss = criterion(qsa_behavior, target)
                loss.backward()
                optimizer.step()

            total_steps += 1
            episode_rewards += reward
            # synchronization
            if total_steps % update_freq == 0:
                target_policy.load_state_dict(behavior_policy.state_dict())
        if episode % 10 == 0:
            print(f"Episode [{episode}] {episode_rewards}")

