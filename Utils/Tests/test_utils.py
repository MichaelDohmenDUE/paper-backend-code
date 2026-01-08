import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from backend.Utils.src.utils import synchronize


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 7)


class TestSynchronize(unittest.TestCase):
    def setUp(self):
        self.source = DummyNet()
        self.target = DummyNet()

        for param in self.source.parameters():
            nn.init.constant_(param, 1.0)
        for param in self.target.parameters():
            nn.init.constant_(param, 0.0)

    def test_hard_update(self):
        synchronize(self.source, self.target, tau=1.0)
        for src, tgt in zip(self.source.parameters(), self.target.parameters()):
            self.assertTrue(torch.allclose(src, tgt))

    def test_soft_update(self):
        tau = 0.5
        original_target = deepcopy(list(self.target.parameters()))
        synchronize(self.source, self.target, tau=tau)
        for src, tgt, orig in zip(self.source.parameters(), self.target.parameters(), original_target):
            expected = tau * src + (1 - tau) * orig
            self.assertTrue(torch.allclose(tgt, expected))

    def test_no_update(self):
        original_target = deepcopy(list(self.target.parameters()))
        synchronize(self.source, self.target, tau=0.0)
        for tgt, orig in zip(self.target.parameters(), original_target):
            self.assertTrue(torch.allclose(tgt, orig))

    def test_invalid_tau(self):
        with self.assertRaises(AssertionError):
            synchronize(self.source, self.target, tau=-0.1)
        with self.assertRaises(AssertionError):
            synchronize(self.source, self.target, tau=1.1)


if __name__ == "__main__":
    unittest.main()
