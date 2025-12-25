import unittest
import numpy as np
from typing import Any

from backend.Utils.src.ReplayBuffer import ReplayBuffer
from backend.Utils.src.BatchTransitioner import TransitionSpec, TransitionFactory

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

        self.make_transition = lambda x: self.factory.create(
            state=np.array([x], dtype=np.float32),
            action=int(x),
            reward=float(x),
            next_state=np.array([x + 1], dtype=np.float32),
            done=False)

    def test_len(self):
        self.assertEqual(len(self.buffer), 0)
        self.buffer.append(self.make_transition(1))
        self.assertEqual(len(self.buffer), 1)

    def test_append(self):
        t = self.make_transition(42)
        self.buffer.append(t)
        self.assertIn(t, self.buffer.buffer)
        self.assertEqual(len(self.buffer), 1)

    def test_extend(self):
        items = [self.make_transition(i) for i in range(3)]
        self.buffer.extend(items)
        self.assertEqual(list(self.buffer.buffer), items)

    def test_sample(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        sample = self.buffer.sample()
        self.assertEqual(len(sample), self.batch_size)
        for item in sample:
            self.assertIn(item, self.buffer.buffer)
    def test_sample_batch(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        batch = self.buffer.sample_batch()
        for field in self.spec.fields:
            self.assertIn(field, batch)
        self.assertEqual(batch["state"].shape[0], self.batch_size)

    def test_sample_sequence(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        seq = self.buffer.sample_sequence_batch(seq_len=2, batch_size=2)
        for field in self.spec.fields:
            self.assertIn(field, seq)
        self.assertEqual(seq["state"].shape[0], 2)
        self.assertEqual(seq["state"].shape[1], 2)

    def test_choice(self):
        items = [self.make_transition(i) for i in range(5)]
        self.buffer.extend(items)
        chosen = self.buffer.choice([0, 2, 4])
        self.assertIn("state", chosen)
        self.assertIn("action", chosen)
        self.assertEqual(chosen["state"].shape[0], 3)
        self.assertEqual(chosen["action"].shape[0], 3)

    def test_buffer_maxlen(self):
        items = [self.make_transition(i) for i in range(6)]
        self.buffer.extend(items)
        self.assertEqual(len(self.buffer), self.buffer_size)
        self.assertNotIn(items[0], self.buffer.buffer)

if __name__ == "__main__":
    unittest.main()
