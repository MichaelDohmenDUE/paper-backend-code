import random
from collections import deque
from typing import Any, Iterable

import numpy as np
import torch

from backend.Utils.src.BatchTransitioner import TransitionBatch, TransitionSpec


class RolloutBuffer:
    def __init__(self, spec: TransitionSpec):
        self.spec = spec
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, x: Any) -> None:
        for f in self.spec.fields:
            if not hasattr(x, f):
                raise ValueError(f"Transition missing field: {f}")
        self.buffer.append(x)

    def clear(self):
        self.buffer = []

    def sample(self) -> dict[str, torch.Tensor]:
        if len(self.buffer) == 0:
            return {}
        batch = {}
        for field in self.spec.fields:
            items = [getattr(t, field) for t in self.buffer]
            if isinstance(items[0], torch.Tensor):
                batch[field] = torch.stack(items)
            elif isinstance(items[0], np.ndarray):
                batch[field] = torch.from_numpy(np.array(items))
            else:
                # Fallback for simple Python floats/ints
                batch[field] = torch.tensor(items)
        self.clear()
        return batch
