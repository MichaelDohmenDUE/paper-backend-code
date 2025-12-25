from copy import deepcopy

import torch

from backend.CommonModels.src.ActorCriticA3C import ActorCritic
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3CDataCollectionProcessor:
    def __init__(self, local_net, env, t_max):
        self.local_net = local_net
        self.env = env
        self.t_max = t_max


    def reset_episode(self):
        self.state = self.env.reset()
        self.done = False

    def run(self):
        pass

class A3CTrainingProcessor:
    def __init__(self, global_net, optimizer, gamma):
        self.global_net = global_net
        self.optimizer = optimizer
        self.gamma = gamma

    def run(self, rollout):
       pass



def main():
    env_name = "CartPole-v1"
    num_workers = 4
    t_max = 5
    gamma = 0.99
    lr = 1e-3
    hidden_size = 64
    seed = 42
    epochs= 1000

    env = EnvironmentHandler(env_name, seed)
    obs_dim, act_dim, _ = env.get_env_specs()
    global_net = ActorCritic(obs_dim, act_dim, hidden_size).to(device)
    optimizer = torch.optim.Adam(global_net.parameters(), lr)

    workers = []
    for wid in range(num_workers):
        local_net = deepcopy(global_net)
        env = EnvironmentHandler(env_name, seed + wid)

        collector = A3CDataCollectionProcessor(local_net, env, t_max)
        trainer = A3CTrainingProcessor(global_net, optimizer, gamma)
        syncer = SyncProcessor(global_net, local_net, tau=1.0, sync_freq=1)

        workers.append((collector, trainer, syncer))

    # Main loop
    for step in range(epochs):
        print(step)
        for collector, trainer, syncer in workers:
            rollout = collector.run()
            trainer.run(rollout)
            syncer.run()

if __name__ == '__main__':
    main()