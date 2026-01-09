import numpy as np

from backend.TD3.src.TD3Trainer import TD3Trainer
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.EvaluationHelper import eval_trainer
from backend.Utils.src.ReplayBuffer import ReplayBuffer


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

    trainer = TD3Trainer(
        state_size=observation_size,
        action_size=action_size,
        hidden_size=hidden_dim,
        max_action=max_action,
        learning_rate=learning_rate,
        tau=tau,
        noise_clip=noise_clip * max_action,
        policy_noise=policy_noise * max_action
    )

    replay_buffer = ReplayBuffer(spec=spec, max_buffer_size=buffer_size, batch_size=batch_size)
    evaluations = [eval_trainer(trainer, env_handler)]

    state = env_handler.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1

        if t < start_timesteps:
            action = np.random.uniform(-max_action, max_action, action_size)
        else:
            noise = np.random.normal(0, max_action * expl_noise, size=action_size)
            action = (trainer.select_action(np.array(state)) + noise).clip(-max_action,max_action)

        next_state, reward, done_env, done_bool =env_handler.step(
            action, episode_timesteps=episode_timesteps
        )

        transition = transition_factory.create(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done_bool,
        )
        replay_buffer.append(transition)

        state = next_state
        episode_reward += reward

        if t >= start_timesteps:
            trainer.train(replay_buffer, batch_size)

        if done_env:
            print(f"Episode {episode_num + 1} — Timestep {t + 1} — Reward: {episode_reward:.2f}")
            state = env_handler.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env_handler, eval_episodes))


if __name__ == "__main__":
    main()
