import torch
from torch.nn import functional as F
import numpy as np
import gymnasium as gym

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.Utils.src.utils import  synchronize
from backend.Utils.src.ReplayBuffer import ReplayBuffer


if __name__ == '__main__':
    env = gym.make("InvertedPendulum-v5")
    action_scale  = float(env.action_space.high[0])
    #env = gym.make("Pendulum-v1")
    observation_size = env.observation_space.shape[0]
    hidden_size = 32
    action_size = env.action_space.shape[0]
    output_size = 1

    num_episodes = 15000
    batch_size = 128

    gamma = 0.99
    lr=5e-3

    # Initialize and synchronize behavior and target networks
    actor = Actor(observation_size, action_size, action_scale, hidden_size)
    actor_target = Actor(observation_size, action_size, action_scale, hidden_size)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    critic = Critic(observation_size, action_size, hidden_size)
    critic_target = Critic(observation_size, action_size, hidden_size)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)


    synchronize(actor, actor_target, 1.0)
    synchronize(critic, critic_target, 1.0)

    buffer = ReplayBuffer()

    for episode in range(num_episodes):

        state, _ = env.reset()
        done = False
        episode_reward = 0
        actor_loss, critic_loss = None, None

        while not done:
            # Data Collection
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = actor(state_tensor)
            action += torch.normal(0, 0.1, size=action.shape)
            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            done = terminated or truncated

            transition = (state, action.item(), reward, next_state, done)
            buffer.append(transition)

            state = next_state
            episode_reward += reward

             # Learning
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
                actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
                rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)

                targets = (rewards_tensor + (1.0 - dones_tensor) * gamma *
                           critic_target(next_states_tensor, actor_target(next_states_tensor))).detach()

                critic_optimizer.zero_grad()
                critic_loss = F.mse_loss(critic(states_tensor, actions_tensor), targets)
                critic_loss.backward()
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                actor_loss = -critic(states_tensor, actor(states_tensor)).mean()
                actor_loss.backward()
                actor_optimizer.step()

                # Synchronization
                synchronize(actor, actor_target, 0.005)
                synchronize(critic, critic_target, 0.005)

        if episode % 10 == 9:
            if actor_loss is not None and critic_loss is not None:
                print(f"Episode: {episode+1}, Reward: {episode_reward}, actor_loss: {actor_loss.mean():.3f}, critic_loss: {critic_loss:.3f}" )
