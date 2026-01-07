
from copy import deepcopy
import torch
from torch import nn

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.DDPG.src.ActionHandler import OUNoise, DeterministicPolicyWithNoise
from backend.SAC.src.DataCollectionProcessor import DataCollectionProcessor
from backend.SAC.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    lr_actor = 1e-4
    lr_critic = 1e-3
    num_episodes = 1000
    env_name = "InvertedPendulum-v5"
    sync_freq = 1
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 0.001
    gamma = 0.99
    seed = 42

    env = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env.get_env_specs()

    # Networks
    actor = Actor(observation_size, action_size, max_action, hidden_size).to(device)
    actor_target = deepcopy(actor).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

    critic = Critic(observation_size, action_size, hidden_size).to(device)
    critic_target = deepcopy(critic).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    noise = OUNoise(action_size, theta=0.15, sigma=0.2, device=device)
    policy = DeterministicPolicyWithNoise(actor, noise, max_action, device)

    data_collection_process = DataCollectionProcessor(env, policy, buffer, factory, device)

    train_process = TrainProcessor(buffer, actor, actor_target, critic, critic_target,actor_optimizer, critic_optimizer, gamma, device)

    sync_process_actor = SyncProcessor(actor, actor_target, tau, sync_freq)
    sync_process_critic = SyncProcessor(critic, critic_target, tau, sync_freq)

    for episode in range(num_episodes):
        done = False
        episode_reward = 0.0
        actor_loss, critic_loss = None, None

        while not done:
            transition = data_collection_process.run()
            actor_loss, critic_loss = train_process.run()
            done = transition.done
            episode_reward += transition.reward

            sync_process_actor.run()
            sync_process_critic.run()

        if episode % 10 == 9 and actor_loss is not None:
            print(
                f"Episode: {episode+1}, Reward: {episode_reward:.2f}, "
                f"actor_loss: {actor_loss.mean():.3f}, critic_loss: {critic_loss:.3f}"
            )

if __name__ == "__main__":
    main()
