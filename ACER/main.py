from backend.ACER.src.ACERTrainer import ACERTrainer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.EvaluationHelper import eval_trainer
from backend.Utils.src.ReplayBuffer import ReplayBuffer

def main():
    env_name = "Reacher-v5"
    seed = 100
    max_timesteps = 1000000
    eval_freq = 2000
    eval_episodes = 10
    batch_size = 32
    learning_rate = 1e-4
    hidden_dim = 256
    tau = 0.01
    buffer_size = int(1e6)
    seq_len = 20
    replay_ratio = 4

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

        action, mu_logp, mu_mean, mu_log_std = trainer.select_action(state, return_params=True)

        next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)

        transition = (state, action, next_state,reward, 1.0 - done_bool,mu_logp, mu_mean, mu_log_std)
        replay_buffer.append(transition)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) >= seq_len:
            trainer.train(replay_buffer, batch_size=1, on_policy=True)

        if len(replay_buffer) >= seq_len:
            for _ in range(replay_ratio):
                trainer.train(replay_buffer, batch_size=batch_size, on_policy=False)

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