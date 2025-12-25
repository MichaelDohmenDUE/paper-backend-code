from typing import Any, Iterable
import random
from collections import deque

from backend.Utils.src.BatchTransitioner import TransitionBatch, TransitionSpec

class ReplayBuffer:
    def __init__(self, spec: TransitionSpec, max_buffer_size: int = 10_000, batch_size: int = 32):
        self.buffer: deque[Any] = deque(maxlen=max_buffer_size)
        self.batch_size: int = batch_size
        self.spec = spec

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
        self.buffer.append(x)

    def extend(self, iterable: Iterable[Any]) -> None:
        for x in iterable:
            self.append(x)

    def sample(self) -> list[Any]:
        return random.sample(self.buffer, self.batch_size)

    def sample_batch(self) -> TransitionBatch:
        transitions = self.sample()
        batch = TransitionBatch(transitions, self.spec)
        return batch.to_tensors()

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
