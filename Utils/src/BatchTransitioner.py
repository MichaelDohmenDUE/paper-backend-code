import numpy as np
import torch
from dataclasses import make_dataclass
import sys

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
    @staticmethod
    def preprocess(data):
        x0 = data[0]
        if isinstance(x0, torch.Tensor):
            return torch.stack(data)
        if isinstance(x0, (int, float, bool, np.integer, np.floating)):
            return torch.tensor(data, dtype=torch.float32)
        if isinstance(x0, np.ndarray) or isinstance(x0, (list, tuple)):
            return torch.tensor(np.array(data), dtype=torch.float32)
        raise ValueError(f"Unknown data type: {type(x0)}")

    def to_tensors(self):
        return {k: self.preprocess(v) for k, v in self.data.items()}
