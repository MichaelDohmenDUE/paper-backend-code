import random
from collections import deque
from typing import Any, Iterable

import torch

from backend.Utils.src.BatchTransitioner import TransitionBatch, TransitionSpec

import numpy as np
import torch
from typing import Any, Iterable
from backend.Utils.src.BatchTransitioner import TransitionSpec


class ReplayBuffer:
    """

    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
    as Inspiration but adapt lazy loading so the Replaybuffer harmonize with the Transitionfactory
    """
    def __init__(self, spec: TransitionSpec, max_buffer_size: int = 10_000, batch_size: int = 32):
        self.spec = spec
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.data: dict[str, np.ndarray] = {}
        self.ptr: int = 0
        self.size: int = 0
        self._initialized: bool = False

    def __len__(self) -> int:
        return self.size

    def _init_buffer(self, first_transition: Any) -> None:
        for field in self.spec.fields:
            if not hasattr(first_transition, field):
                raise ValueError(f"Transition missing field: {field}")
            val = np.array(getattr(first_transition, field))
            self.data[field] = np.zeros((self.max_buffer_size, *val.shape), dtype=val.dtype)

        self._initialized = True

    def append(self, x: Any) -> None:
        if not self._initialized:
            self._init_buffer(x)
        for field in self.spec.fields:
            self.data[field][self.ptr] = getattr(x, field)

        self.ptr = (self.ptr + 1) % self.max_buffer_size
        self.size = min(self.size + 1, self.max_buffer_size)

    def extend(self, iterable: Iterable[Any]) -> None:
        for x in iterable:
            self.append(x)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return self.choice(idxs)

    def sample_sequence_batch(self, seq_len: int, batch_size: int) -> dict[str, torch.Tensor]:
        if seq_len > self.size:
            raise BufferError(f"Buffersize {self.size}) < {seq_len} .")

        valid_starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
        seq_idxs = valid_starts[:, None] + np.arange(seq_len)

        return self.choice(seq_idxs)

    def choice(self, indices: np.ndarray | list[int]) -> dict[str, torch.Tensor]:
        batch = {}
        for field in self.spec.fields:
            numpy_batch = self.data[field][indices]

            batch[field] = torch.as_tensor(numpy_batch)

        return batch