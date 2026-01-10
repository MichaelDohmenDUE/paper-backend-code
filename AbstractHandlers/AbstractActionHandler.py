from abc import ABC, abstractmethod


class AbstractActionHandler(ABC):
    @abstractmethod
    def select_action(self, state):
        pass
