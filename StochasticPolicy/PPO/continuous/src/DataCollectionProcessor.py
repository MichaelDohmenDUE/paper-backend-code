from backend.AbstractHandlers.AbstractActionHandler import AbstractActionHandler
from backend.Utils.src import RolloutBuffer
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler
from backend.Utils.src.ReplayBuffer import ReplayBuffer


def bootstaping_value(replay_buffer: ReplayBuffer, done: bool, policy, state):
    if done:
        last_value = 0.0
    else:
        _, _, last_value = policy.select_action(state)
    replay_buffer.buffer[-1].bootstrap_value = last_value

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
        self.state = self.env_handler.reset()

    def run(self):
        episode_timesteps = 0
        episodic_reward = 0
        metrics = {}

        self.replay_buffer.buffer.clear()

        for _ in range(self.rollout_size):
            action, logp, value = self.policy.select_action(self.state)
            next_state, reward, done, info = self.env_handler.step(action)
            transition = self.transition_factory.forward(state=self.state, action=action, logp=logp, reward=reward,
                                                         done=done, value=value, bootstrap_value=None
                                                         )
            self.replay_buffer.append(transition)
            episodic_reward += reward
            episode_timesteps += 1
            self.total_steps += 1
            self.state = reset_handler(self.env_handler, next_state, done)
            ###Logging
            if done:
                metrics = {"charts/episodic_return": episodic_reward,
                           "charts/episodic_length": episode_timesteps,
                           "global_step": self.total_steps
                           }
                episode_timesteps = 0
                episodic_reward = 0
            ##### Logging End
        bootstaping_value(self.replay_buffer, done, self.policy, self.state)
        return metrics
