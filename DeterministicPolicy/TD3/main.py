import copy

import torch
from torch import optim

from backend.CommonModels.src.Actor import Actor
from backend.CommonModels.src.Critic import Critic
from backend.DeterministicPolicy.TD3.src.ActionHandler import ActionHandler
from backend.DeterministicPolicy.TD3.src.DataCollectionProcessor import DataCollectionProcessor
from backend.DeterministicPolicy.TD3.src.TD3TrainerProcessor import TrainProcessor
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.EvaluationHelper import eval_trainer
from backend.Utils.src.GlobalCounter import GlobalCounter
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.SyncProcessor import SyncProcessor

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
    sync_freq = 2
    learning_rate = 3e-4
    tau = 0.005
    noise_clip = 0.5
    policy_noise = 0.2
    hidden_dim = 256
    buffer_size = int(1e6)

    env_handler = EnvironmentHandler(env_name, seed)
    observation_size, action_size, max_action = env_handler.get_env_specs()
    if max_action is None:
        max_action = 1
    spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])

    transition_factory = TransitionFactory(spec)

    actor = Actor(observation_size, action_size, max_action, hidden_dim).to(device)

    actor_target = copy.deepcopy(actor).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)

    critic_1 = Critic(observation_size, action_size, hidden_dim).to(device)
    critic_2 = Critic(observation_size, action_size, hidden_dim).to(device)
    critic_target_1 = copy.deepcopy(critic_1).to(device)
    critic_target_2 = copy.deepcopy(critic_2).to(device)

    optimizer_critic_1 = optim.Adam(critic_1.parameters(), lr=learning_rate)
    optimizer_critic_2 = optim.Adam(critic_2.parameters(), lr=learning_rate)

    action_handler = ActionHandler(actor, action_size, max_action, expl_noise, noise_clip, start_timesteps, device)
    replay_buffer = ReplayBuffer(spec=spec, max_buffer_size=buffer_size, batch_size=batch_size)

    gl_counter = GlobalCounter()

    trainer = TrainProcessor(
        actor=actor,
        actor_target=actor_target,
        critic_1=critic_1,
        critic_2=critic_2,
        critic_target_1=critic_target_1,
        critic_target_2=critic_target_2,
        optimizer_critic_1=optimizer_critic_1,
        optimizer_critic_2=optimizer_critic_2,
        optimizer_actor=optimizer_actor,
        replay_buffer=replay_buffer,
        global_counter=gl_counter,
        max_action=max_action,
        learning_rate=learning_rate,
        start_timesteps=start_timesteps,
        synchro_frequency=sync_freq,
        noise_clip=noise_clip * max_action,
        policy_noise=policy_noise * max_action,
        device=device,
    )

    datacollector = DataCollectionProcessor(env_handler, action_handler, transition_factory, replay_buffer, gl_counter)

    sync_process_critic_1 = SyncProcessor(critic_1, critic_target_1, tau, sync_freq, gl_counter)
    sync_process_critic_2 = SyncProcessor(critic_2, critic_target_2, tau, sync_freq, gl_counter)
    sync_process_actor = SyncProcessor(actor, actor_target, tau, sync_freq, gl_counter)

    evaluations = [eval_trainer(trainer, env_handler)]

    for t in range(max_timesteps):
        datacollector.run()
        trainer.run()
        sync_process_critic_1.run()
        sync_process_critic_2.run()
        sync_process_actor.run()
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env_handler, eval_episodes))


if __name__ == "__main__":
    main()
