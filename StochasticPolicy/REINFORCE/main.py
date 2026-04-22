import time

import wandb

from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.StochasticPolicy.REINFORCE.src.ActionHandler import ActionHandler
import torch

from backend.CommonModels.src.Policy_Reinforce import PolicyVPG
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.StochasticPolicy.REINFORCE.src.DataCollector import DataCollectionProcessor
from backend.StochasticPolicy.REINFORCE.src.ReinforceTrainer import REINFORCETrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Related Paper: https://link.springer.com/article/10.1007/BF00992696
    """
    start_time = time.time()
    learn_rate = 1e-5
    max_steps = 1000
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

    spec = TransitionSpec(["logp", "reward", "done"])
    transition_factory = TransitionFactory(spec)
    replay_buffer = RolloutBuffer(spec)

    gym_factory = GymEnvFactory(env_name)
    env_handler = EnvironmentHandler(gym_factory, seed=seed)

    observation_size, action_size, _ = env_handler.get_env_specs()
    policy = PolicyVPG(observation_size, action_size, hidden_dim=hidden_dim).to(device)

    action_handler = ActionHandler(policy, device)
    optimizer = torch.optim.SGD(policy.parameters(), lr=learn_rate)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, replay_buffer, policy, device)

    trainer = REINFORCETrainer(replay_buffer, optimizer, beta, gamma, device=device)

    for step in range(max_steps):
        data_collector.run()
        metrics = trainer.run()
        if metrics and step % 400 == 0:
            metrics["charts/SPS"] = int(step / (time.time() - start_time))
            wandb.log(metrics, step=step)
            #TODO: THINK ABOUT AN EVAL FOR LATER / also doesn't log
       # if step % 1_000 == 0 and episode > 0:
        #    avg_score = evaluate_policy(behavior_net, eval_env, episodes=10, device=device)
        #    wandb.log({"charts/eval_avg_score": avg_score}, step=step)

if __name__ == "__main__":
    main()
