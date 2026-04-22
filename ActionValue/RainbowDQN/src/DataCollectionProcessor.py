import torch
import wandb
from torch import nn

from backend.ActionValue.RainbowDQN.src.ActionHandler import GreedyPolicy
from backend.Utils.src.BatchTransitioner import TransitionFactory
from backend.Utils.src.EnviromentHandler import EnvironmentHandler
from backend.Utils.src.NodeLib.NodeLibrary import reset_handler
from backend.Utils.src.PrioReplayBuffer import PrioReplayBuffer
from backend.Utils.src.StepBuffer import StepBuffer


class DataCollectionProcessor:
    def __init__(self, policy: nn.Module, env: EnvironmentHandler, buffer: PrioReplayBuffer, step_buffer: StepBuffer,
                 greedy: GreedyPolicy, transition_factory: TransitionFactory, device: torch.device,
                 v_min: float, v_max: float, atoms: int) -> None:
        self.policy = policy
        self.env = env
        self.step_buffer = step_buffer
        self.replay_buffer = buffer
        self.state = env.reset()
        self.done = False
        self.greedy = greedy
        self.transition_factory = transition_factory
        self.device = device
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.support = torch.linspace(v_min, v_max, atoms, device=self.device)
        # Logging
        self.episode_count = 0
        self.episode_reward = 0.0
        self.total_steps = 0
        self.episode_steps = 0

    def run(self):
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.policy.reset_noise()
            logits = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            q_values = (probs * self.support).sum(dim=-1).squeeze(0)
        action = self.greedy.select_action(q_values=q_values)
        next_state, reward, done, done_bool = self.env.step(action.item())
        transition = self.transition_factory.forward(state=self.state, action=action.item(), reward=reward,
                                                     next_state=next_state, done=done)
        n_step_transition = self.step_buffer.push(transition)
        if n_step_transition is not None:
            self.replay_buffer.append(n_step_transition)
        self.state = reset_handler(env=self.env, next_state=next_state, done=done)
        if done:
            for t in self.step_buffer.flush_transitions():
                self.replay_buffer.append(t)
        # Logging
        self.episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
        if done:
            if self.episode_count % 10 == 0:
                print(f"Episode [{self.episode_count}] {self.episode_reward}")
            try:
                wandb.log({
                    "charts/episodic_return": self.episode_reward,
                    "charts/episodic_length": self.episode_steps,
                    "global_step": self.total_steps,
                })
            except Exception as e:
                print(f"Logging error: {e}")
            self.episode_count += 1
            self.episode_reward = 0.0
            self.episode_steps = 0