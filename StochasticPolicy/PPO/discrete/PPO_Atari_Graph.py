import time

import torch
import wandb
from torch import optim

from backend.CommonModels.src.DiscreteActorPPO import AtariPPOAgent
from backend.StochasticPolicy.PPO.discrete.src.DataCollectorGraph import DataCollectionProcessor
from backend.StochasticPolicy.PPO.discrete.src.DiscreteActionHandlerAtari import ActionHandler
from backend.StochasticPolicy.PPO.discrete.src.PPOTrainerAtari import PPOTrainerProcessor
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.BatchTransitioner import TransitionSpec
from backend.Utils.src.EnvFactory import AtariEnvFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.RolloutBuffer import RolloutBuffer
from backend.Utils.src.utils import setting_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.uint8, device=device)
            state_t = state_t.float()
            with torch.no_grad():
                dist, value = trainer.actor(state_t)
                action = dist.probs.argmax(dim=-1).cpu().numpy()
            next_state, reward, done, _ = env_handler.step(action)
            avg_reward += reward[0]
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main(seed):
    start_time = time.time()
    env_name = "PongNoFrameskip-v4"
    seed = seed
    rollout_size = 2048
    batch_size = 256
    epochs = 4
    max_steps = 10_000_000
    lr = 2.5e-4
    gamma = 0.99
    lam = 0.95
    eval_freq = 100_000
    eval_episodes = 5
    algo_name = "PPO_discrete_atari"
    num_envs = 8
    offset = 100
    setting_global_seed(seed)

    wandb.init(
        project="my-ppo-benchmarks",
        group=algo_name,
        name=f"{algo_name}-seed-{seed}",
        tags=[env_name, "debugging", algo_name],
        reinit=True,
        entity="michael_dohmen-",
        config={
            "env_id": env_name,
            "exp_name": "my_ppo_Pong",
            "seed": seed,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "max_steps": max_steps,
            "offset": offset,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
        }
    )

    spec = TransitionSpec(["state", "action", "logp", "reward", "done", "value", "bootstrap_value"])
    transition_factory = TransitionFactory(spec)
    factory = AtariEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(factory, seed, num_envs)
    eval_env_handler = VecEnvironmentHandler(factory, seed + offset, 1)
    state_dim, action_dim, _ = env_handler.get_env_specs()
    channels = state_dim[0]
    agent = AtariPPOAgent(action_dim, channels).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    rollout_buffer = RolloutBuffer(spec, rollout_size)

    trainer = PPOTrainerProcessor(agent, optimizer, rollout_buffer, batch_size, epochs, gamma=gamma, lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, rollout_buffer, rollout_size,
                                             agent, device)
    steps = 0
    eval_step = 0
    while steps < max_steps:
        metrics_ep = data_collector.run()
        metrics_train = trainer.run()
        all_metrics = {**metrics_ep, **metrics_train, "charts/SPS": int(steps / (time.time() - start_time)),
                       "global_step": data_collector.context["total_steps"]}
        steps = data_collector.context["total_steps"]

        if eval_step <= steps:
            avg_eval_reward = eval_trainer(trainer, eval_env_handler, eval_episodes=eval_episodes)
            all_metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        wandb.log(all_metrics, step=steps)
    wandb.finish()

if __name__ == "__main__":
    seed = [0,1,2]
    for seed in seed:
        main(seed)
