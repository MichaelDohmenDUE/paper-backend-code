from backend.TD3.src.DataCollectionProcessor import DataCollectionProcessor
import numpy as np
import torch

from backend.CommonModels.src.Actor import Actor
from backend.TD3.src.ActionHandler import ActionHandler
from backend.TD3.src.TD3Trainer import TD3Trainer
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.EvaluationHelper import eval_trainer
from backend.Utils.src.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env_name = "HalfCheetah-v5"
    seed = 100
    max_timesteps = 1000000
    start_timesteps = 25000
    eval_freq = 2000
    eval_episodes = 10
    expl_noise = 0.1
    batch_size = 256
    learning_rate = 3e-4
    tau = 0.005
    noise_clip = 0.5
    policy_noise = 0.2
    hidden_dim = 256
    buffer_size = int(1e6)

    env_handler = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env_handler.get_env_specs()

    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])

    transition_factory = TransitionFactory(spec)

    actor = Actor(observation_size, action_size, max_action, hidden_dim).to(device)

    trainer = TD3Trainer(
        actor = actor,
        state_size=observation_size,
        action_size=action_size,
        hidden_size=hidden_dim,
        max_action=max_action,
        learning_rate=learning_rate,
        tau=tau,
        noise_clip=noise_clip * max_action,
        policy_noise=policy_noise * max_action
    )

    action_handler = ActionHandler(actor, action_size, max_action, expl_noise, start_timesteps,device)

    replay_buffer = ReplayBuffer(spec=spec, max_buffer_size=buffer_size, batch_size=batch_size)
    evaluations = [eval_trainer(trainer, env_handler)]

    datacollector = DataCollectionProcessor(env_handler, action_handler, transition_factory, replay_buffer)

    state = env_handler.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        datacollector.run()
        if t >= start_timesteps:
            trainer.train(replay_buffer)
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env_handler, eval_episodes))


if __name__ == "__main__":
    main()
