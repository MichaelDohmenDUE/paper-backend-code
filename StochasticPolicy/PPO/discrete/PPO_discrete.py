import time

import torch
import wandb
from mujoco.rollout import Rollout
from torch import optim

from backend.CommonModels.src.CriticPPO import CriticPPO
from backend.CommonModels.src.DiscreteActorPPO import DiscreteActorPPO
from backend.StochasticPolicy.PPO.continuous.src.DataCollectionProcessor import DataCollectionProcessor
from backend.StochasticPolicy.PPO.discrete.src.DiscreteActionHandler import ActionHandler
from backend.StochasticPolicy.PPO.discrete.src.PPOTrainerProcessor import PPOTrainerProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnvFactory import GymEnvFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state.reshape(1, -1)).to(trainer.device)
            with torch.no_grad():
                dist = trainer.actor(state_t)
                action = dist.probs.argmax(dim=-1).item()
            next_state, reward, done, _ = env_handler.step(action)
            avg_reward += reward
            state = next_state

    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main():
    start_time = time.time()
    env_name = "CartPole-v1"
    seed = 4
    rollout_size = 2048
    batch_size = 32
    epochs = 10
    max_steps = 1_000_000
    lr = 3e-4
    hidden_dim = 64
    gamma = 0.99
    lam = 0.95
    eval_freq = 10000
    setting_global_seed(seed)

    wandb.init(
        entity="michael_dohmen-",
        project="my-ppo-benchmarks",
        config={
            "env_id": env_name,
            "exp_name": "my_ppo_cartpole",
            "seed": seed,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "max_steps": max_steps,
        }
    )

    spec = TransitionSpec(["state", "action", "logp", "reward", "done", "value", "bootstrap_value"])
    transition_factory = TransitionFactory(spec)
    factory = GymEnvFactory(env_name)
    env_handler = EnvironmentHandler(factory, seed)
    eval_env_handler = EnvironmentHandler(factory, seed + 100)
    state_dim, action_dim, _ = env_handler.get_env_specs()

    actor = DiscreteActorPPO(state_dim, action_dim, hidden_dim).to(device)
    critic = CriticPPO(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    action_handler = ActionHandler(actor, critic, device)

    rollout_buffer = RolloutBuffer(spec)

    trainer = PPOTrainerProcessor(actor, critic, optimizer, rollout_buffer, batch_size, epochs, gamma=gamma, lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, rollout_buffer, rollout_size,
                                             action_handler)
    steps = 0
    eval_step = 0
    while steps < max_steps:
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(steps / (time.time() - start_time)),
                       "global_step": data_collector.total_steps}
        steps = data_collector.total_steps

        if eval_step <= steps:
            avg_eval_reward = eval_trainer(trainer, eval_env_handler, eval_episodes=10)
            all_metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        wandb.log(all_metrics, step=steps)


if __name__ == "__main__":
    main()
