import numpy as np

from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler
from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler


def bootstraping_value(done: bool, policy, state):
    if done:
        last_value = 0.0
    else:
        _, _, last_value = policy.select_action(state)
    return last_value


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, transition_factory: TransitionFactory,
                 replay_buffer: RolloutBuffer.RolloutBuffer, rollout_size: int,
                 action_handler: AbstractActionHandler):
        self.env_handler = env_handler
        self.transition_factory = transition_factory
        self.replay_buffer = replay_buffer
        self.rollout_size = rollout_size
        self.policy = action_handler
        self.total_steps = 0
        self.num_envs = env_handler.num_envs
        self.current_rewards = np.zeros(self.num_envs)
        self.state = self.env_handler.reset().astype(np.uint8)

    def run(self):
        metrics = {}
        finished_episode_rewards = []
        self.replay_buffer.buffer.clear()
        num_collection_steps = self.rollout_size // self.num_envs
        for _ in range(num_collection_steps):
            actions, logps, values = self.policy.select_action(self.state)
            next_states, rewards, dones, infos = self.env_handler.step(actions)
            self.current_rewards += rewards
            clipped_rewards = np.clip(rewards, -1, 1)
            for i in range(self.num_envs):
                transition = self.transition_factory.forward(
                    state=self.state[i],
                    action=actions[i],
                    logp=logps[i],
                    reward=clipped_rewards[i],
                    done=dones[i],
                    value=values[i],
                    bootstrap_value=0.0
                )
                self.replay_buffer.append(transition)

                self.state[i] = reset_handler(self.env_handler, next_states, dones[i]).astype(np.uint8)
                if dones[i]:
                    finished_episode_rewards.append(self.current_rewards[i])
                    self.current_rewards[i] = 0  # Reset for next game
            self.state = next_states.astype(np.uint8)
            self.total_steps += self.num_envs
        _, _, final_values = self.policy.select_action(self.state)

        for i in range(self.num_envs):
            boot_val = 0.0 if dones[i] else final_values[i]
            self.replay_buffer.buffer[-(self.num_envs - i)].bootstrap_value = boot_val
        if finished_episode_rewards:
            metrics = {"charts/episodic_return": np.mean(finished_episode_rewards),
                       "global_step": self.total_steps}
        return metrics
