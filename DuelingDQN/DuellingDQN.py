import torch
from torch import nn
import random
import gymnasium as gym
import numpy as np

from backend.CommonModels.src.DuellingDQN import DuellingDQN
from backend.Utils.src.utils import  synchronize
import torch.nn.functional as F
from backend.Utils.src.ReplayBuffer import ReplayBuffer

def epsilon_greedy(q_values: torch.tensor, epsilon: float) -> torch.tensor:
    """ Epsilon-greedy policy, returns random action if random number is < epsilon, else greedy action
        Randomly samples from max actions if there is a tie.
     """
    actions = torch.arange(len(q_values))
    max_q_value = torch.max(q_values)
    max_idx = (q_values == max_q_value).to(torch.int64)
    greedy_action = random.choice(actions[max_idx == 1])

    return random.choice(actions) if random.random() < epsilon else greedy_action

def get_epsilon(frame_idx: int) -> float:
    frac = min(1.0, frame_idx / epsilon_decay_frames)
    return epsilon_start + (epsilon_final - epsilon_start) * frac


if __name__ == '__main__':

    ## SEEDING TODO: Extract later
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")
    observation_size = env.observation_space.shape[0]
    hidden_size = 32
    action_size = env.action_space.n
    env = gym.make("CartPole-v1")

    behavior_policy = DuellingDQN(observation_size, hidden_size, action_size).to(device)

    target_policy = DuellingDQN(observation_size, hidden_size, action_size).to(device)

    target_policy.load_state_dict(behavior_policy.state_dict())
    lr = 1e-3
    optimizer = torch.optim.Adam(behavior_policy.parameters(), lr=lr)


    buffer = ReplayBuffer()

    gamma = 0.99
    num_episodes = 600
    batch_size = 64
    replay_start_size = 1000
    update_freq = 1000
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay_frames = 50000.0
    max_grad_norm = 10.0
    clipping = True


    # Outer training loop
    total_steps = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        episode_rewards = 0
        while not done:

            # Data collection
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # 1 x obs
            q_values = behavior_policy(state_tensor).squeeze(0)  # action_dim
            eps = get_epsilon(total_steps)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(q_values.argmax().item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.append((state, action, reward, next_state, float(done)))
            state = next_state


            if len(buffer) > max(batch_size, replay_start_size):
                transitions = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*transitions)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

                qsa_behavior = behavior_policy(states_tensor).gather(1, actions_tensor)

                with torch.no_grad():
                    a_max = behavior_policy(next_states_tensor).argmax(dim=1, keepdim=True)
                    q_next = target_policy(next_states_tensor).gather(1, a_max)
                    target = rewards_tensor + gamma * q_next * (1.0 - dones_tensor)

                loss = F.smooth_l1_loss(qsa_behavior, target)
                optimizer.zero_grad()
                loss.backward()
                if clipping:
                    torch.nn.utils.clip_grad_norm_(behavior_policy.parameters(), max_grad_norm)
                optimizer.step()

            episode_rewards += reward
            total_steps += 1
            # synchronization
            if total_steps % update_freq == 0:
                synchronize(behavior_policy, target_policy, tau=1.0)
        if episode % 10 == 0:
            print(f"Episode [{episode}] {episode_rewards}")
