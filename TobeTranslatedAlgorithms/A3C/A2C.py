import torch
from torch import optim

from TobeTranslatedAlgorithms.A3C.src.A3CDataCollectionProcessor import A3CDataCollectionProcessor
from TobeTranslatedAlgorithms.A3C.src.A3CTrainingProcessor import A3CTrainingProcessor
from backend.CommonModels.src.ActorCriticA3C import ActorCritic
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.SyncProcessor import SyncProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2CWorker:
    def __init__(self, worker_id, env_name, seed, t_max, gamma, global_net, optimizer, factory, obs_dim,
                 act_dim, hidden):
        torch.manual_seed(seed + worker_id)

        self.local_net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.local_net.load_state_dict(global_net.state_dict())
        self.global_net = global_net
        self.optimizer = optimizer
        gym_factory = GymEnvFactory(env_name) # TODO: clean this up with the call later
        self.env = EnvironmentHandler(gym_factory, seed + worker_id)

        self.collector = A3CDataCollectionProcessor(
            self.local_net, self.env, t_max, factory, gamma
        )

        self.trainer = A3CTrainingProcessor(
            self.global_net, self.local_net, self.optimizer,
            gamma, entropy_coef=0.001, max_grad_norm=40.0
        )

        self.syncer = SyncProcessor(self.global_net, self.local_net, tau=1.0, sync_freq=1)

    def collector_process(self):
        self.syncer.run()
        rollout = self.collector.run()
        return rollout

    def train_step(self, rollout):
        self.trainer.run(rollout)

def main():
    env_name = "CartPole-v1"
    num_workers = 4
    t_max = 20
    gamma = 0.99
    lr = 7e-4
    hidden_size = 64
    seed = 42
    num_updates = 1000000

    torch.manual_seed(seed)
    gym_factory = GymEnvFactory(env_name)
    env = EnvironmentHandler(gym_factory, seed)
    obs_dim, act_dim, _ = env.get_env_specs()
    global_net = ActorCritic(obs_dim, act_dim, hidden_size).to(device)
    optimizer = optim.RMSprop(global_net.parameters(), lr=lr)

    spec_fields = ["state", "action", "reward", "value", "log_prob", "done", "entropy"]
    spec = TransitionSpec(spec_fields)
    factory = TransitionFactory(spec)

    workers = [A2CWorker(i, env_name, seed, t_max, gamma, global_net, optimizer,
                         factory, obs_dim, act_dim, hidden_size) for i in range(num_workers)]
    for step in range(num_updates):
        rollouts = [w.collector_process() for w in workers]
        for w, rollout in zip(workers, rollouts):
            w.train_step(rollout)


if __name__ == "__main__":
    main()
