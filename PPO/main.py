from backend.PPO.src.PPO_continuous import PPOTrainer
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.utils import compute_gae


def eval_trainer(trainer, env_handler, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env_handler.reset()
        done = False
        while not done:
            action, _, _ = trainer.select_action(state)
            next_state, reward, done, _ = env_handler.step(action, 0)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

def collect_rollout(env_handler, trainer, replay_buffer, rollout_size, episode_timesteps):
    state = env_handler.reset()
    for step in range(rollout_size):
        episode_timesteps += 1
        action, logp, value = trainer.select_action(state)
        next_state, reward, done, done_bool = env_handler.step(action, episode_timesteps)
        replay_buffer.append((state, action, logp, reward, done, value))
        if not done:
            state = next_state
        else:
            env_handler.reset()
        if done:
            episode_timesteps = 0
    _, _, last_value = trainer.select_action(state)
    return last_value


def train_update(trainer, replay_buffer, last_value, batch_size, epochs):
    states, actions, logps, advs, rets = compute_gae(replay_buffer, gamma=0.99, lam=0.95, last_value=last_value)
    stats = trainer.train(states, actions, logps, advs, rets,batch_size=batch_size, epochs=epochs)
    replay_buffer.buffer.clear()
    return stats

def main():
    env_name = "HalfCheetah-v5"
    seed = 100
    rollout_size = 2048
    batch_size = 64
    epochs = 10
    num_updates = 1000

    env_handler = EnvironmentHandler(env_name, seed)
    trainer = PPOTrainer(
        state_dim=env_handler.state_dim,
        action_dim=env_handler.action_dim,
        hidden_dim=64,
        lr=3e-4
    )
    replay_buffer = ReplayBuffer(buffer_size=rollout_size)
    episode_timesteps = 0

    for update in range(num_updates):
        last_value = collect_rollout(env_handler, trainer, replay_buffer, rollout_size, episode_timesteps)

        train_update(trainer, replay_buffer, last_value, batch_size, epochs)

        if update % 10 == 0:
            eval_trainer(trainer, env_handler, eval_episodes=5)

if __name__ == "__main__":
    main()
