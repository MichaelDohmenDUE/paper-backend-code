from copy import deepcopy

import torch
from torch import nn

from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.DQN.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DQN.src.TrainProcessor import TrainProcessor
from backend.DQN.src.TrainProcessorGraph import TrainProcessor
from backend.DQN.src.dqn_graph import build_dqn_graph
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.Node import Graph
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # initialization
    lr = 1e-3
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 32
    batch_size = 64
    max_buffer_size = 10000
    tau = 1.0
    gamma = 0.99
    max_steps = 100000
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)
    seed = 42

    env = EnvironmentHandler(env_name, seed)
    obs_size, action_size, max_action = env.get_env_specs()

    behavior_net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_size)).to(
        device)
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    target_net = deepcopy(behavior_net).to(device)

    buffer = ReplayBuffer(spec, max_buffer_size, batch_size)
    collector = DataCollectionProcessor(behavior_net, env, buffer, EpsilonGreedyPolicy(epsilon), factory, device)
    dqn_graph = Graph(build_dqn_graph())
    train_process = TrainProcessor(dqn_graph, buffer, behavior_net, target_net, optimizer, gamma, device)
    # train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device)
    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == '__main__':
    main()
