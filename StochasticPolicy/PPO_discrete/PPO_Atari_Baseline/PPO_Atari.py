import time

import torch
import wandb
from torch import optim

from backend.Utils.src.ReplayBuffer import ReplayBuffer
from StochasticPolicy.PPO_discrete_Mujoco.src.DiscreteActorPPO import AtariPPOAgent
from StochasticPolicy.PPO_discrete.PPO_Atari_Baseline.src.DataCollectorAtari import DataCollectionProcessor
from StochasticPolicy.PPO_discrete.PPO_Atari_Baseline.src.PPOTrainerAtari import PPOTrainerProcessor
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
                dist, value = trainer.agent(state_t)
                action = dist.probs.argmax(dim=-1).cpu().numpy()
            next_state, reward, done, _ = env_handler.step(action)
            avg_reward += reward[0]
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main(seed):
    start_time = time.time()
    env_name = "BreakoutNoFrameskip-v4"
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
        name=f"{algo_name}-{env_name}-seed-{seed}",
        tags=[env_name, "benching", algo_name],
        reinit=True,
        entity="michael_dohmen-",
        config={
            "env_id": env_name,
            "exp_name": f"PPO_{env_name}_seed-{seed}",
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
            "device": device
        }
    )

    spec = TransitionSpec(["state", "action", "logp", "reward", "done", "value", "bootstrap_value"])
    replay_spec = TransitionSpec(["state", "action", "logp", "advantage", "return"])
    transition_factory = TransitionFactory(spec)
    factory = AtariEnvFactory(env_name)
    env_handler = VecEnvironmentHandler(factory, seed, num_envs)
    eval_env_handler = VecEnvironmentHandler(factory, seed + offset, 1)
    state_dim, action_dim, _ = env_handler.get_env_specs()
    channels = state_dim[0]
    agent = AtariPPOAgent(action_dim, channels).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    rollout_buffer = RolloutBuffer(spec, rollout_size)
    replay_buffer = ReplayBuffer(replay_spec, rollout_size, batch_size)

    trainer = PPOTrainerProcessor(agent, optimizer, rollout_buffer, replay_buffer, batch_size, epochs, gamma=gamma,
                                  lam=lam)

    data_collector = DataCollectionProcessor(env_handler, transition_factory, rollout_buffer, rollout_size,
                                             agent, device)
    step = 0
    eval_step = 0
    while step < max_steps:
        metrics = {}
        metrics_ep = data_collector.run()
        if metrics_ep:
            metrics.update(metrics_ep)
        metrics_train = trainer.run()
        if metrics_train:
            metrics.update(metrics_train)
        step = data_collector.context["total_steps"]
        metrics["global_step"] = step

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            sps = int(step / elapsed_time)
            metrics["charts/SPS"] = sps
        if eval_step <= step:
            avg_eval_reward = eval_trainer(trainer, eval_env_handler, eval_episodes=eval_episodes)
            metrics["eval/avg_reward"] = avg_eval_reward
            eval_step += eval_freq
        if len(metrics) > 1:
            wandb.log(metrics, step=step)
    # Final Eval
    metrics = {"global_step": step,
               "charts/eval_avg_score": eval_trainer(trainer, eval_env_handler, eval_episodes=eval_episodes)}
    wandb.log(metrics, step=step)
    wandb.finish()
if __name__ == "__main__":
    seeds = [2,1,0]
    for seed in seeds:
        main(seed)
