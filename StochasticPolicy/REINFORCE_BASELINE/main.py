import time

import wandb

from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.RolloutBuffer import RolloutBuffer
import torch

from backend.CommonModels.src.Policy_Reinforce_Baseline import PolicyReinforceBaseline
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.StochasticPolicy.REINFORCE_BASELINE.src.DataCollector import DataCollectionProcessor
from backend.StochasticPolicy.REINFORCE_BASELINE.src.ReinforceTrainer import REINFORCETrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Related Paper: https://link.springer.com/article/10.1007/BF00992696
    """
    start_time = time.time()
    learn_rate = 1e-5
    max_steps = 10000
    seed = 1
    hidden_dim = 64
    env_name = "CartPole-v1"
    beta = 0.01
    gamma = 0.99

    wandb.init(
        entity="michael_dohmen-",
        project="my-REINFORCE-benchmarks",
        config={
            "env_id": env_name,
            "exp_name": "REINFORCE-CartPole-v1",
            "seed": seed,
            "lr": learn_rate,
            "gamma": gamma,
            "max_steps": max_steps,
            "hidden_dim": hidden_dim,
        }
    )

    spec = TransitionSpec(["state", "logp", "reward", "done"])
    transition_factory = TransitionFactory(spec)
    replay_buffer = RolloutBuffer(spec)

    gym_factory = GymEnvFactory(env_name)
    env_handler = EnvironmentHandler(gym_factory, seed=seed)

    observation_size, action_size, _ = env_handler.get_env_specs()
    policy = PolicyReinforceBaseline(observation_size, action_size, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, policy, device)

    trainer = REINFORCETrainer(replay_buffer, policy, optimizer, beta, gamma, device=device)

    for step in range(max_steps):
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(step / (time.time() - start_time)),
                       "global_step": data_collector.total_steps}
        wandb.log(all_metrics, step=step)
        # TODO: What about Eval?
if __name__ == "__main__":
    main()
