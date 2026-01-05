from copy import deepcopy
import torch

from backend.CommonModels.src.Policy import Policy
from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.DDQN.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DDQN.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 1.0
    gamma = 0.99
    max_steps = 100000
    seed = 42
    lr = 5e-4

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    env = EnvironmentHandler(env_name, seed)
    obs_size, action_size, _ = env.get_env_specs()

    behavior_net = Policy(obs_size, action_size, hidden_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)

    collector = DataCollectionProcessor(behavior_net, env, buffer,EpsilonGreedyPolicy(epsilon), factory, device)

    train_process = TrainProcessor(buffer, behavior_net, target_net,optimizer, gamma, device)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == "__main__":
    main()