

from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.TD3.src.ActionHandler import ActionHandler
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.ReplayBuffer import ReplayBuffer


class DataCollectionProcessor:
    def __init__(self, env_handler: EnvironmentHandler, action_handler: ActionHandler,
                 transition_factory: TransitionFactory, replay_buffer: ReplayBuffer):
        self.env = env_handler
        self.action_handler = action_handler
        self.factory = transition_factory
        self.buffer = replay_buffer

        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0
        self.global_timestep = 0

    def run(self):
        action = self.action_handler.select_action(self.state, self.global_timestep)
        self.episode_timesteps += 1
        next_state, reward, done_env, done_bool = self.env.step(action, episode_timesteps=self.episode_timesteps)

        transition = self.factory.create(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done_bool,
        )

        self.buffer.append(transition)
        self.state = next_state
        self.episode_reward += reward
        if done_env:
            print(f"Episode {self.episode_num + 1} — Reward: {self.episode_reward:.2f}")
            self.state = self.env.reset()
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_num += 1

        self.global_timestep += 1

        return transition
