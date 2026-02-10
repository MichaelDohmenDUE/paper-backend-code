import torch

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.DPG.src.ActionHandler import OUNoise, DeterministicPolicyWithNoise
from backend.DPG.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DPG.src.TrainProcessor import TrainProcess
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    lr_actor = 1e-4
    lr_critic = 1e-3
    max_timesteps = 20000
    env_name = "InvertedPendulum-v5"
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    gamma = 0.99
    seed = 42

    env = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env.get_env_specs()

    # Networks
    actor = Actor(observation_size, action_size, max_action, hidden_size).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

    critic = Critic(observation_size, action_size, hidden_size).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    noise = OUNoise(action_size, theta=0.15, sigma=0.2, device=device)
    policy = DeterministicPolicyWithNoise(actor, noise, max_action, device)

    gl_counter = GlobalCounter()

    data_collection_process = DataCollectionProcessor(env, policy, buffer, factory, gl_counter, device)
    train_process = TrainProcess(buffer, actor, critic, actor_optimizer, critic_optimizer,gamma, device)
    for t in range(max_timesteps):
        data_collection_process.run()
        train_process.run()

if __name__ == "__main__":
    main()
