import torch

class Node:
    def __init__(self, name, function, inputs, outputs, no_grad=False):
        self.name = name
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        self.no_grad = no_grad

    def __call__(self, context):
        args = [context[key] for key in self.inputs]
        if self.no_grad:
            with torch.no_grad():
                result = self.function(*args)
        else:
            result = self.function(*args)

        if len(self.outputs) == 1:
            context[self.outputs[0]] = result
        else:
            for key, value in zip(self.outputs, result):
                context[key] = value

class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
    def run(self, ctx: dict):
        for node in self.nodes:
            node(ctx)
        return ctx