from copy import deepcopy

import torch

from backend.CommonModels.src.DuellingDQN import DuellingDQN
from backend.CommonModels.src.RainbowDuellingDQN import RainbowDuellingDQN
from backend.DQN.src.ActionHandler import EpsilonGreedyPolicy
from backend.RainbowDQN.src.DataCollectionProcessor import DataCollectionProcessor
from backend.RainbowDQN.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.PrioReplayBuffer import PrioReplayBuffer
from backend.Utils.src.StepBuffer import StepBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    epsilon = 0.2
    env_name = "CartPole-v1"
    sync_freq = 40
    hidden_size = 128
    batch_size = 64
    atoms_size = 51
    max_buffer_size = 1000000
    tau = 1.0
    gamma = 0.99
    max_steps = 100000
    seed = 42
    lr = 5e-4
    max_grad_norm = 10.0

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
    factory = TransitionFactory(spec)

    env = EnvironmentHandler(env_name, seed)
    obs_size, action_size, _ = env.get_env_specs()

    behavior_net = RainbowDuellingDQN(obs_size, hidden_size, action_size, atoms_size).to(device)
    target_net = deepcopy(behavior_net).to(device)

    optimizer = torch.optim.Adam(behavior_net.parameters(), lr)

    step_buffer = StepBuffer(factory, lookahead_n=3, gamma=gamma)
    buffer = PrioReplayBuffer(spec, max_buffer_size, batch_size)

    collector = DataCollectionProcessor(behavior_net, env, buffer, step_buffer, EpsilonGreedyPolicy(epsilon), factory,
                                        device)

    train_process = TrainProcessor(buffer, behavior_net, target_net, optimizer, gamma, device, max_grad_norm)

    sync_process = SyncProcessor(behavior_net, target_net, tau, sync_freq)

    for step in range(max_steps):
        collector.run()
        train_process.run()
        sync_process.run()


if __name__ == "__main__":
    main()
