import unittest

import numpy as np
import torch

from Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory
from Utils.src.ReplayBuffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.spec = TransitionSpec(["state", "action", "reward", "next_state", "done"])
        self.factory = TransitionFactory(self.spec)
        self.buffer_size = 5
        self.batch_size = 3
        self.buffer = ReplayBuffer(
            spec=self.spec,
            max_buffer_size=self.buffer_size,
            batch_size=self.batch_size
        )

        self.make_transition = lambda x: self.factory.forward(
            state=np.array([x], dtype=np.float32),
            action=int(x),
            reward=float(x),
            next_state=np.array([x + 1], dtype=np.float32),
            done=False
        )

    def test_len(self):
        self.assertEqual(len(self.buffer), 0)
        self.buffer.append(self.make_transition(1))
        self.assertEqual(len(self.buffer), 1)

    def test_append(self):
        t = self.make_transition(42)
        self.buffer.append(t)
        self.assertEqual(len(self.buffer), 1)

        np.testing.assert_array_equal(self.buffer.data["state"][0], t.state)
        self.assertEqual(self.buffer.data["reward"][0], t.reward)

    def test_extend(self):
        items = [self.make_transition(i) for i in range(3)]
        self.buffer.extend(items)
        self.assertEqual(len(self.buffer), 3)

        np.testing.assert_array_equal(self.buffer.data["state"][2], items[2].state)

    def test_sample_batch(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        batch = self.buffer.sample_batch()

        self.assertIsInstance(batch["state"], torch.Tensor)

        for field in self.spec.fields:
            self.assertIn(field, batch)

        self.assertEqual(batch["state"].shape[0], self.batch_size)

    def test_sample_sequence(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        seq = self.buffer.sample_sequence_batch(seq_len=2, batch_size=2)

        for field in self.spec.fields:
            self.assertIn(field, seq)

        # Shape should be (batch_size, seq_len, feature_shape)
        self.assertEqual(seq["state"].shape[0], 2)  # batch_size
        self.assertEqual(seq["state"].shape[1], 2)  # seq_len

    def test_choice(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        chosen = self.buffer.choice([0, 2, 4])

        self.assertIn("state", chosen)
        self.assertIn("action", chosen)
        self.assertEqual(chosen["state"].shape[0], 3)
        expected_states = torch.tensor([[0.], [2.], [4.]], dtype=torch.float32)
        self.assertTrue(torch.equal(chosen["state"], expected_states))

    def test_buffer_maxlen_and_overwrite(self):
        items = [self.make_transition(i) for i in range(6)]
        self.buffer.extend(items)

        self.assertEqual(len(self.buffer), self.buffer_size)

        self.assertEqual(self.buffer.ptr, 1)

        np.testing.assert_array_equal(self.buffer.data["state"][0], items[-1].state)


if __name__ == "__main__":
    unittest.main()