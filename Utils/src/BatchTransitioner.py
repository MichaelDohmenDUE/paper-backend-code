import sys
from dataclasses import make_dataclass
from typing import Any

import numpy as np
import torch


class TransitionSpec:
    def __init__(self, fields):
        self.fields = fields


class TransitionFactory:
    def __init__(self, spec: TransitionSpec):
        self.spec = spec

        cls = make_dataclass("Transition", [(f, object) for f in spec.fields])

        module = sys.modules[__name__]
        setattr(module, "Transition", cls)
        self.transition_cls = cls

    def create(self, **kwargs):
        for f in self.spec.fields:
            if f not in kwargs:
                raise ValueError(f"Missing field: {f}")
        return self.transition_cls(**kwargs)


class TransitionBatch:
    def __init__(self, transitions, spec: TransitionSpec):
        self.fields = spec.fields
        self.data = {
            f: [getattr(t, f) for t in transitions]
            for f in self.fields
        }

    def __getitem__(self, key) -> list[Any]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def preprocess(data):
        x0 = data[0]

        # Convert to tensor
        if isinstance(x0, torch.Tensor):
            out = torch.stack(data)
        elif isinstance(x0, (int, float, bool, np.integer, np.floating)):
            out = torch.tensor(data, dtype=torch.float32)
        elif isinstance(x0, np.ndarray) or isinstance(x0, (list, tuple)):
            out = torch.tensor(np.array(data), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown data type: {type(x0)}")

        # Shape normalization
        if out.dim() == 1:
            out = out.unsqueeze(-1)

        return out

    def to_tensors(self):
        return {k: self.preprocess(v) for k, v in self.data.items()}

    def unpack(self):
        return tuple(self.to_tensors()[field] for field in self.fields)