from collections import deque
from typing import List

from backend.Utils.src.BatchTransitioner import TransitionFactory, TransitionSpec


class StepBuffer(object):
    def __init__(self, transition_factory: TransitionFactory, lookahead_n: int = 3, gamma: float = 1.0):
        self.transition_factory = transition_factory
        self.lookahead_n = lookahead_n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, transition) -> TransitionSpec | None:
        self.buffer.append(transition)

        if len(self.buffer) < self.lookahead_n:
            return None
        else:
            return_in_n_steps = 0.0
            done = False
            final_next_state = None
            for i in range(self.lookahead_n):
                transition = self.buffer[i]
                return_in_n_steps += (self.gamma ** i) * transition.reward
                if transition.done:
                    done = transition.done
                    final_next_state = transition.next_state
                    break
            # Bootstrap last state in case it is  empty
            if final_next_state is None:
                final_next_state = self.buffer[self.lookahead_n - 1].next_state

            first_transition = self.buffer[0]

            transition_n_steps = self.transition_factory.forward(state=first_transition.state,
                                                                 action=first_transition.action,
                                                                 reward=return_in_n_steps, next_state=final_next_state,
                                                                 done=done)

            self.buffer.popleft()

            return transition_n_steps

    def flush_transitions(self) -> List[TransitionSpec]:
        out = []
        while len(self.buffer) > 0:
            return_in_n_steps = 0.0
            done = False
            final_next_state = None

            for i, transition in enumerate(self.buffer):
                return_in_n_steps += (self.gamma ** i) * transition.reward
                if transition.done:
                    done = True
                    final_next_state = transition.next_state
                    break
                if i + 1 >= self.lookahead_n:
                    break

            if final_next_state is None:
                final_next_state = self.buffer[min(self.lookahead_n - 1, len(self.buffer) - 1)].next_state

            first = self.buffer[0]

            transition = self.transition_factory.forward(state=first.state, action=first.action,
                                                         reward=return_in_n_steps, next_state=final_next_state,
                                                         done=done)

            out.append(transition)
            self.buffer.popleft()

        return out
