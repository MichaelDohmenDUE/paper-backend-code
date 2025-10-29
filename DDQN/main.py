import torch
from torch import nn
import random
import numpy as np

from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.utils import  synchronize
import torch.nn.functional as F
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy(q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
    """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
        Randomly samples from max actions if there is a tie.
     """
    actions = torch.arange(len(q_values), device=q_values.device)
    max_q_value = torch.max(q_values)
    max_idx = (q_values == max_q_value).to(torch.bool)
    greedy_action = random.choice(actions[max_idx].tolist())
    return random.choice(actions.tolist()) if random.random() < epsilon else greedy_action


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_handler = EnvironmentHandler(env_name="CartPole-v1", seed=0)

    observation_size = env_handler.state_dim
    action_size = env_handler.action_dim
    hidden_size = 32

    behavior_policy = nn.Sequential(
        nn.Linear(observation_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size)
    ).to(device)

    target_policy = nn.Sequential(
        nn.Linear(observation_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size)
    ).to(device)

    target_policy.load_state_dict(behavior_policy.state_dict())

    optimizer = torch.optim.Adam(behavior_policy.parameters())
    buffer = ReplayBuffer()

    gamma = 0.99
    num_episodes = 600
    update_freq = 40
    batch_size = 64

    total_steps = 0
    for episode in range(num_episodes):
        state = env_handler.reset()
        done = False
        episode_timesteps = 0
        episode_rewards = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = behavior_policy(state_tensor)
            action = epsilon_greedy(q_values=q_values, epsilon=0.2)

            next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_timesteps += 1
            episode_rewards += reward

            if len(buffer) > batch_size:
                transitions = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*transitions)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)

                qsa_behavior = behavior_policy(states_tensor).gather(1, actions_tensor)

                next_qs_behavior = behavior_policy(next_states_tensor)
                next_actions = torch.argmax(next_qs_behavior, dim=1, keepdim=True)

                next_qs_target = target_policy(next_states_tensor)
                qsa_target = next_qs_target.gather(1, next_actions).detach()

                target = rewards_tensor + gamma * qsa_target * (1.0 - dones_tensor)

                optimizer.zero_grad()
                loss = F.mse_loss(qsa_behavior, target)
                loss.backward()
                optimizer.step()

            total_steps += 1
            if total_steps % update_freq == 0:
                synchronize(behavior_policy, target_policy, tau=1.0)

        if episode % 10 == 0:
            print(f"Episode [{episode}] Reward: {episode_rewards}")
