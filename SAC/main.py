
from copy import deepcopy
import torch

from backend.CommonModels.src.ActorSAC import ActorSAC as Actor
from backend.CommonModels.src.Critic import Critic
from backend.SAC.src.ActionSelector import ActionSelector
from backend.SAC.src.DataCollectionProcessor import DataCollectionProcessor
from backend.SAC.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    num_episodes = 1000
    env_name = "HalfCheetah-v5"
    sync_freq = 1
    hidden_size = 256
    batch_size = 256
    max_buffer_size = 100000
    tau = 0.005
    gamma = 0.99
    seed = 42

    env = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env.get_env_specs()

    # Networks
    actor = Actor(observation_size, action_size, max_action, hidden_size).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    critic_1 = Critic(observation_size, action_size, hidden_size).to(device)
    critic_target_1 = deepcopy(critic_1).to(device)
    critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=lr_critic)

    critic_2 = Critic(observation_size, action_size, hidden_size).to(device)
    critic_target_2 = deepcopy(critic_2).to(device)
    critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=lr_critic)
    # temperature and entropy
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr_alpha)
    target_entropy = -action_size

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    policy = ActionSelector(actor ,device)


    data_collection_process = DataCollectionProcessor(env, policy, buffer, factory, device)

    train_process = TrainProcessor(buffer, actor, critic_1, critic_target_1, critic_2, critic_target_2 ,actor_optimizer, critic_optimizer_1, critic_optimizer_2,
                                   log_alpha, alpha_optimizer, target_entropy, gamma, device)

    sync_process_critic_1 = SyncProcessor(critic_1, critic_target_1, tau, sync_freq)
    sync_process_critic_2 = SyncProcessor(critic_2, critic_target_2, tau, sync_freq)
    total_steps = 0
    episode = 0
    while total_steps < 1000000:
        done = False
        episode_reward = 0.0
        actor_loss, critic_loss_1, critic_loss_2, alpha_loss = None, None, None, None

        while not done:
            transition = data_collection_process.run()
            total_steps += 1
            actor_loss, critic_loss_1, critic_loss_2, alpha_loss = train_process.run()
            done = data_collection_process.done
            episode_reward += transition.reward

            sync_process_critic_1.run()
            sync_process_critic_2.run()
        episode += 1
        if episode % 10 == 9 and actor_loss is not None:
            print(
                f"Episode: {episode+1}, Steps: {total_steps},  Reward: {episode_reward:.2f}, "
                f"actor_loss: {actor_loss:.3f}, critic_loss_1: {critic_loss_1:.3f}"
                f"critic_loss_2: {critic_loss_2:.3f}, alpha_loss: {alpha_loss:.3f}"
            )

if __name__ == "__main__":
    main()
