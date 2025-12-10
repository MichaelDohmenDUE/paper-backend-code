import numpy as np
from backend.ACER.src.ACERTrainer import ACERTrainer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            action , _= trainer.select_action(state)
            next_state, reward, done, _ = env_handler.step(action, 0)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main():
    env_name = "InvertedPendulum-v5"
    seed = 100
    max_timesteps = 1000000
    start_timesteps = 25000
    eval_freq = 2000
    eval_episodes = 10
    batch_size = 256
    learning_rate = 3e-4
    hidden_dim = 256
    tau = 1.0
    buffer_size = int(1e6)

    env_handler = EnvironmentHandler(env_name, seed)

    trainer = ACERTrainer(
        state_size=env_handler.state_dim,
        action_size=env_handler.action_dim,
        hidden_size=hidden_dim,
        learning_rate=learning_rate,
        gamma=0.99,
        tau=tau,
        trust_region_delta=0.01
    )

    replay_buffer = ReplayBuffer(buffer_size=buffer_size)
    evaluations = [eval_trainer(trainer, env_handler)]

    state = env_handler.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1


        action, mu_logp = trainer.select_action(np.array(state))

        next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)
        replay_buffer.append((state, action, next_state, reward, 1.0 - done_bool, mu_logp.item()))


        state = next_state
        episode_reward += reward

        if len(replay_buffer) >= batch_size:
            trainer.train(replay_buffer, batch_size)

        if done:
            print(f"Episode {episode_num+1} — Timestep {t+1} — Reward: {episode_reward:.2f}")
            state = env_handler.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_trainer(trainer, env_handler, eval_episodes))


if __name__ == "__main__":
    main()