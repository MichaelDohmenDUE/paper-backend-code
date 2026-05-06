from copy import deepcopy

import torch
import wandb

from TobeTranslatedAlgorithms.SAC.src.ActorSAC import ActorSAC as Actor
from backend.CommonModels.src.Critic import Critic
from TobeTranslatedAlgorithms.SAC.src.ActionSelector import ActionSelector
from TobeTranslatedAlgorithms.SAC.src.DataCollectionProcessor import DataCollectionProcessor
from TobeTranslatedAlgorithms.SAC.src.TrainProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    max_steps = 1_000_000
    env_name = "HalfCheetah-v4"
    sync_freq = 1
    hidden_size = 256
    batch_size = 256
    max_buffer_size = 100000
    tau = 0.005
    gamma = 0.99
    seed = 42
    algo_name = "SAC"
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="my-SAC-benchmarks",
        name=f"{algo_name}-seed-{seed}",
        tags=[env_name, "testing", algo_name],
        config={
            "env_id": env_name,
            "exp_name": f"{algo_name}-{env_name}",
            "seed": seed,
            "max_buffer_size": max_buffer_size,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "learning_rate_actor": lr_actor,
            "learning_rate_critic": lr_critic,
            "learning_rate_alpha": lr_alpha,
            "sync_freq": sync_freq,
            "max_steps": max_steps,
            "gamma": gamma,
            "tau": tau,
        }
    )

    gym_factory = GymEnvFactory(env_name)

    env = EnvironmentHandler(gym_factory, seed)
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

    policy = ActionSelector(actor, device)

    data_collection_process = DataCollectionProcessor(env, policy, buffer, factory, device)

    train_process = TrainProcessor(buffer, actor, critic_1, critic_target_1, critic_2, critic_target_2, actor_optimizer,
                                   critic_optimizer_1, critic_optimizer_2,
                                   log_alpha, alpha_optimizer, target_entropy, gamma, device)

    sync_process_critic_1 = SyncProcessor(critic_1, critic_target_1, tau, sync_freq)
    sync_process_critic_2 = SyncProcessor(critic_2, critic_target_2, tau, sync_freq)
    total_steps = 0
    while total_steps < max_steps:
        total_steps += 1
        metrics_dl = data_collection_process.run()
        metrics_train = train_process.run()

        sync_process_critic_1.run()
        sync_process_critic_2.run()

        all_metrics = metrics_dl | metrics_train

        # TODO: Eval is missing right now

        wandb.log(all_metrics, step=total_steps)
    wandb.finish()


if __name__ == "__main__":
    main()
