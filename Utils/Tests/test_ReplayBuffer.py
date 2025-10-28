import unittest

from backend.Utils.src.ReplayBuffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer_size = 5
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size)

    def test_len(self):
        self.assertEqual(len(self.buffer), 0)
        self.buffer.append(1)
        self.assertEqual(len(self.buffer), 1)

    def test_str(self):
        self.buffer.extend([1, 2, 3])
        self.assertEqual(str(self.buffer), str([1, 2, 3]))

    def test_append(self):
        self.buffer.append("Michael")
        self.assertIn("Michael", self.buffer.buffer)
        self.assertEqual(len(self.buffer), 1)

    def test_extend(self):
        items = [1, 2, 3]
        self.buffer.extend(items)
        self.assertEqual(list(self.buffer.buffer), items)

    def test_sample(self):
        self.buffer.extend([10, 20, 30, 40, 50])
        sample = self.buffer.sample(3)
        self.assertEqual(len(sample), 3)
        for item in sample:
            self.assertIn(item, self.buffer.buffer)

    def test_choice(self):
        self.buffer.extend(["a", "b", "c", "d", "e"])
        indices = [0, 2, 4]
        chosen = self.buffer.choice(indices)
        expected = ["a", "c", "e"]
        self.assertEqual(chosen, expected)

    def test_buffer_maxlen(self):
        self.buffer.extend([1, 2, 3, 4, 5, 6])
        self.assertEqual(len(self.buffer), self.buffer_size)
        self.assertNotIn(1, self.buffer.buffer)

if __name__ == "__main__":
    unittest.main()
