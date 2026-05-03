import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_trainer(trainer, env_handler, eval_episodes=5):
    trainer.actor.eval()

    episode_returns = []
    current_episode_reward = 0.0

    state = env_handler.reset()
    if isinstance(state, tuple):
        state = state[0]
    while len(episode_returns) < eval_episodes:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            action = trainer.actor(state_t).cpu().numpy()

        next_state, reward, done, _ = env_handler.step(action)
        r = reward[0] if isinstance(reward, np.ndarray) else reward
        current_episode_reward += r
        if done[0]:
            episode_returns.append(current_episode_reward)
            current_episode_reward = 0.0

        state = next_state

    trainer.actor.train()

    metrics = {
        "eval/eval_avg_reward": float(np.mean(episode_returns)),
    }

    return metrics
