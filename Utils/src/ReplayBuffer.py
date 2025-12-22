from typing import Iterable, Any
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size: int = 10_000):
        self.buffer: deque[Any] = deque(maxlen=buffer_size)

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
        self.buffer.append(x)

    def extend(self, iterable: Iterable[Any]) -> None:
        self.buffer.extend(iterable)

    def sample(self, batch_size: int) -> list[Any]:
        return random.sample(self.buffer, batch_size)

    def sample_sequence(self, seq_len: int, batch_size: int) -> list[list[Any]]:
        length_buffer = len(self.buffer)
        if seq_len > length_buffer:
            raise BufferError(f"Buffer (Size {length_buffer}) is not long enough to allow a {seq_len}")
        indices = [random.randint(0, length_buffer - seq_len) for _ in range(batch_size)]
        sequences = []
        for idx in indices:
            sequence = [self.buffer[idx + t] for t in range(seq_len)]
            sequences.append(sequence)
        return sequences

    def choice(self, indices : list[Any]) -> list[Any]:
        indexed_list = [self.buffer[idx] for idx in indices]
        return indexed_list
