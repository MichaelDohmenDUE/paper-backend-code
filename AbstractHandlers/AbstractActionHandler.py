from abc import ABC, abstractmethod
from typing import Any


class AbstractActionHandler(ABC):
    @abstractmethod
    def select_action(self, state) -> tuple[Any, ...]:
        pass
