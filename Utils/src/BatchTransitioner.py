import numpy as np
import torch
from dataclasses import make_dataclass


class TransitionSpec:
    def __init__(self, fields):
        self.fields = fields

class TransitionFactory:
    def __init__(self, spec: TransitionSpec):
        self.spec = spec
        self.transition_cls = make_dataclass(
            "Transition",
            [(f, object) for f in spec.fields])

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
    @staticmethod
    def preprocess(data):
        x0 = data[0]
        if isinstance(x0, (int, np.integer)):
            return torch.tensor(data, dtype=torch.int64).unsqueeze(-1)
        elif isinstance(x0, (bool, float, np.floating)):
            return torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        elif isinstance(x0, np.ndarray):
            return torch.tensor(np.array(data), dtype=torch.float32)
        elif isinstance(x0, (list, tuple)):
            return torch.tensor(np.array(data), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown data type: {type(x0)}")

    def to_tensors(self):
        return {k: self.preprocess(v) for k, v in self.data.items()}
