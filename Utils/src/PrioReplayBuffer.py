import random
from collections import deque
from typing import Any, Iterable

import torch

from backend.Utils.src.BatchTransitioner import TransitionBatch, TransitionSpec
from backend.Utils.src.SumSegmentTree import SumSegmentTree, MinSegmentTree


class PrioReplayBuffer:
    def __init__(self, spec: TransitionSpec,
                 max_buffer_size: int = 10_000,
                 batch_size: int = 32,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 1e-6,
                 ):
        self.buffer: deque[Any] = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.batch_size: int = batch_size
        self.spec = spec
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.idx = 0

        capacity = 1
        while capacity < max_buffer_size:
            capacity *= 2
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def __str__(self) -> str:
        return f"{list(self.buffer)}"

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self.buffer)[idx]
        else:
            return self.buffer[idx]

    def append(self, x: Any) -> None:
        for f in self.spec.fields:
            if not hasattr(x, f):
                raise ValueError(f"Transition missing field: {f}")

        idx = self.idx
        self.idx = (self.idx + 1) % self.max_buffer_size
        if len(self.buffer) < self.max_buffer_size:
            self.buffer.append(x)
        else:
            self.buffer[idx] = x
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

    def extend(self, iterable: Iterable[Any]) -> None:
        for x in iterable:
            self.append(x)

    def sample(self) -> tuple[list[Any], list[Any]]:
        total = self.sum_tree.total()
        segment = total / self.batch_size
        indices: list[int] = []
        transitions: list[list[Any]] = []

        for i in range(self.batch_size):
            s: float = random.uniform(segment * i, segment * (i + 1))
            idx = self.sum_tree.retrieve(s)
            indices.append(idx)
            transitions.append(self.buffer[idx])

        return transitions, indices

    def sample_batch(self) -> dict[str, torch.Tensor]:
        transitions, indices = self.sample()

        total = self.sum_tree.total()
        probs = torch.tensor([self.sum_tree.tree[idx + self.sum_tree.capacity - 1] / total for idx in indices],
                             dtype=torch.float32)

        weights = (len(self.buffer) * probs) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = TransitionBatch(transitions, self.spec)
        return {
            "BatchTensor": batch.to_tensors(),
            "Indices": indices,
            "Weights": weights
        }

    def sample_sequence_batch(self, seq_len: int, batch_size: int):
        length_buffer = len(self.buffer)
        if seq_len > length_buffer:
            raise BufferError(f"Buffer (Size {length_buffer}) is not long enough to allow a {seq_len}")
        indices = [random.randint(0, length_buffer - seq_len) for _ in range(batch_size)]
        sequences = []
        for idx in indices:
            seq = [self.buffer[idx + t] for t in range(seq_len)]
            sequences.append(seq)
        batched = {}
        for field in self.spec.fields:
            field_values = [[getattr(t, field) for t in seq] for seq in sequences]
            batched[field] = TransitionBatch.preprocess(field_values)
        return batched

    def choice(self, indices: list[int]):
        transitions = [self.buffer[idx] for idx in indices]
        batch = TransitionBatch(transitions, self.spec)
        return batch.to_tensors()

    def update_priorities(self, indices, priorities) -> None:
        for idx, p in zip(indices, priorities):
            p = (abs(p) + 1e-6) ** self.alpha
            self.sum_tree.update(idx, p)
            self.min_tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
