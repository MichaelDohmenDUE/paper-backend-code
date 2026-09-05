import random
from collections import deque
from typing import Any, Iterable

import numpy as np
import torch

from Utils.src.BatchTransitioner import TransitionBatch, TransitionSpec


class RolloutBuffer:
    def __init__(self, spec: TransitionSpec, rollout_size: int):
        self.spec = spec
        self.num_envs = 8
        self.buffer = []
        self.rollout_size = rollout_size

    def __len__(self) -> int:
        return len(self.buffer)

    def reached_rollout_size(self) -> bool:
        return len(self.buffer) >= self.rollout_size

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


class KStepRolloutBuffer(RolloutBuffer):
    def __init__(self, spec, rollout_size: int):
        super().__init__(spec, rollout_size)
        self.ready = False

    def is_ready(self) -> bool:
        return self.ready

    def set_ready(self, state: bool) -> None:
        self.ready = state

    def clear(self):
        super().clear()
        self.ready = False