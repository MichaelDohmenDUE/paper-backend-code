import random

import numpy as np
import torch

from backend.DDQN.src.DDQNTrainer import DDQNTrainer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.CommonModels.src.Policy import Policy


def epsilon_greedy(q_values: torch.Tensor, epsilon: float) -> int:
    q_values = q_values.to("cpu")
    actions = torch.arange(len(q_values))
    max_q_value = torch.max(q_values)
    max_idx = (q_values == max_q_value).to(torch.bool)
    greedy_action = random.choice(actions[max_idx].tolist())
    return random.choice(actions.tolist()) if random.random() < epsilon else greedy_action


def main():
    env_name = "CartPole-v1"
    seed = 0
    num_episodes = 600
    batch_size = 64
    update_freq = 40
    hidden_size = 32
    gamma = 0.99
    epsilon = 0.2
    tau = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_handler = EnvironmentHandler(env_name, seed)

    behavior_policy = Policy(env_handler.state_dim, env_handler.action_dim, hidden_size).to(device)
    target_policy = Policy(env_handler.state_dim, env_handler.action_dim, hidden_size).to(device)
    target_policy.load_state_dict(behavior_policy.state_dict())

    optimizer = torch.optim.Adam(behavior_policy.parameters())
    buffer = ReplayBuffer()

    trainer = DDQNTrainer(
        env_handler=env_handler,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        optimizer=optimizer,
        buffer=buffer,
        gamma=gamma,
        batch_size=batch_size,
        update_freq=update_freq,
        epsilon=epsilon,
        tau=tau,
        device=device
    )

    for episode in range(num_episodes):
        state = env_handler.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = behavior_policy(state_tensor)
            action = epsilon_greedy(q_values, epsilon)

            next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_timesteps += 1
            episode_reward += reward

            trainer.train()

        if episode % 10 == 0:
            print(f"Episode [{episode}] Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()