from collections import deque

import numpy as np
import torch

from TobeTranslatedAlgorithms.ACER.src.discreteAtari.ACERTrainer import ACERTrainer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import VecEnvironmentHandler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class ACERDataCollector:
    def __init__(self, trainer: ACERTrainer, env: VecEnvironmentHandler, buffer: ReplayBuffer,
                 factory: TransitionFactory,
                 device: torch.device, seq_len: int = 20):
        self.trainer = trainer
        self.env = env
        self.buffer = buffer
        self.factory = factory
        self.device = device
        self.seq_len = seq_len
        self.num_envs = getattr(env, "num_envs", 1)

        self.rollouts = [[] for _ in range(self.num_envs)]
        self.state = np.array(env.reset(), dtype=np.uint8)
        ##LLL
        self.running_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.recent_scores = deque(maxlen=100)  # Tracks the last 100 completed episodes
        self.episodes_completed = 0
        #LLM End
    def run(self):
        state_t = torch.from_numpy(self.state).float().to(self.device)
        with torch.no_grad():
            logits, _ = self.trainer.model(state_t)
            logits = torch.clamp(logits, -20, 20)
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = dist.sample()
            logp_tensor = dist.log_prob(action_tensor)

        action = action_tensor.cpu().numpy()
        mu_logp = logp_tensor.cpu().numpy()
        mu_logits = logits.cpu().numpy()

        next_state_raw, reward, done, info = self.env.step(action)
        self.running_rewards += reward
        ### LLM for Debugging
        for i in range(self.num_envs):
            if done[i]:
                self.recent_scores.append(self.running_rewards[i])
                self.episodes_completed += 1
                self.running_rewards[i] = 0.0
                if self.episodes_completed % 20 == 0:
                    mean_score = np.mean(self.recent_scores)
                    print(
                        f"[TRAIN ROLLOUT] Episodes: {self.episodes_completed} | Recent 100-ep Mean Score: {mean_score:.2f}")
        ### LLM END
        next_state_raw = np.array(next_state_raw, dtype=np.uint8)
        clipped_reward = np.clip(reward, -1, 1)

        true_next_state = next_state_raw.copy()
        if isinstance(info, dict) and "final_observation" in info:
            final_obs = info["final_observation"]
            for i, d in enumerate(done):
                if d and final_obs[i] is not None:
                    true_next_state[i] = final_obs[i]

        for i in range(self.num_envs):
            tr = self.factory.forward(
                state=self.state[i],
                action=np.int16(action[i]),
                reward=np.int8(clipped_reward[i]),
                next_state=true_next_state[i],
                mask=np.uint8(1 - done[i]),
                mu_logp=np.float16(mu_logp[i]),
                mu_logits=mu_logits[i].astype(np.float16)
            )
            self.rollouts[i].append(tr)

        self.state = next_state_raw

        if len(self.rollouts[0]) == self.seq_len:
            completed_rollouts = self.rollouts
            for env_rollout in completed_rollouts:
                for tr in env_rollout:
                    self.buffer.append(tr)

            self.rollouts = [[] for _ in range(self.num_envs)]
            return completed_rollouts

        return None
