import unittest
from unittest.mock import MagicMock

import torch

from backend.CommonModels.src.Actor import Actor
from backend.DeterministicPolicy.TD3.src.TD3TrainerProcessor import TD3Trainer


class TestTD3Trainer(unittest.TestCase):
    def setUp(self):
        self.state_size = 3
        self.action_size = 2
        self.hidden_size = 64
        self.max_action = 1.0
        self.learning_rate = 0.001
        self.tau = 0.005
        self.noise_clip = 0.2
        self.policy_noise = 0.1
        self.batch_size = 32

        self.trainer = TD3Trainer(
            actor=Actor(action_size=self.action_size, state_size=self.state_size, max_action=self.max_action,
                        hidden_size=self.hidden_size),
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            max_action=self.max_action,
            learning_rate=self.learning_rate,
            tau=self.tau,
            noise_clip=self.noise_clip,
            policy_noise=self.policy_noise
        )

        self.mock_buffer = MagicMock()

        def fake_sample_batch():
            state = torch.randn(self.batch_size, self.state_size)
            action = torch.randn(self.batch_size, self.action_size)
            reward = torch.randn(self.batch_size, 1)
            next_state = torch.randn(self.batch_size, self.state_size)
            done = torch.randint(0, 2, (self.batch_size, 1)).float()
            return {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }

        self.mock_buffer.sample_batch.side_effect = fake_sample_batch

    def test_actor_update(self):
        self.trainer.iteration = self.trainer.syncro_frequency - 1

        initial_actor_weights = [p.clone() for p in self.trainer.actor.parameters()]
        initial_target_weights = [p.clone() for p in self.trainer.actor_target.parameters()]

        self.trainer.train(self.mock_buffer)

        updated_actor_weights = [p for p in self.trainer.actor.parameters()]
        self.assertFalse(all(torch.equal(i, u) for i, u in zip(initial_actor_weights, updated_actor_weights)))

    def test_updates_critics(self):
        initial_critic_1_weights = [p.clone() for p in self.trainer.critic_1.parameters()]
        initial_critic_2_weights = [p.clone() for p in self.trainer.critic_2.parameters()]

        self.trainer.train(self.mock_buffer)

        updated_critic_1_weights = [p for p in self.trainer.critic_1.parameters()]
        updated_critic_2_weights = [p for p in self.trainer.critic_2.parameters()]

        self.assertFalse(all(torch.equal(i, u) for i, u in zip(initial_critic_1_weights, updated_critic_1_weights)))
        self.assertFalse(all(torch.equal(i, u) for i, u in zip(initial_critic_2_weights, updated_critic_2_weights)))
