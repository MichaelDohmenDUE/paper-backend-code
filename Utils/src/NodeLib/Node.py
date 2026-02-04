from collections import deque

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
        if len(self.outputs) == 0:
            return
        if len(self.outputs) == 1:
            context[self.outputs[0]] = result
        else:
            for key, value in zip(self.outputs, result):
                context[key] = value


class Graph:
    def __init__(self, nodes):
        self.nodes: list[Node] = nodes

    def run(self, context: dict):
        queue = deque(self.nodes)
        executed_notes = set()
        while len(queue) > 0:
            current_node = queue.popleft()

            if all(input_ in context for input_ in current_node.inputs):
                current_node(context)
                executed_notes.add(current_node)
            else:
                queue.append(current_node)
        return context

    def validate(self, start_context: dict) -> bool:
        produced = set(start_context.keys())
        all_outputs = set()

        # Check for Duplicate Node Outputs
        for node in self.nodes:
            for output in node.outputs:
                #if output in all_outputs:
                #   print(f"Warning: '{output}' overwritten by node {node.name}")
                all_outputs.add(output)
        # Check for missing inputs
        for node in self.nodes:
            for input in node.inputs:
                if input not in produced and input not in all_outputs:
                    raise ValueError(f"Input '{input}' for node '{node.name}' is never produced")

        remaining = set(self.nodes)
        resolved = set(produced)
        # LOOP detection
        progress: bool = True
        while progress and remaining:
            progress = False
            for node in list(remaining):
                if all(input_ in resolved for input_ in node.inputs):
                    resolved.update(node.outputs)
                    remaining.remove(node)
                    progress = True

        if remaining:
            names = [node.name for node in remaining]
            raise ValueError(f"Cycle or unresolved inputs: {names}")

        return True
