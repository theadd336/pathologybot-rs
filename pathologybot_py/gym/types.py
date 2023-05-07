from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EnvState:
    state: np.ndarray
    reward: float
    is_final: bool


class Gym(ABC):
    @abstractmethod
    def reset(self) -> EnvState:
        pass

    @abstractmethod
    def step(self, action: int) -> EnvState:
        pass

    @abstractmethod
    def max_state_value(self) -> float:
        pass


class InvalidActionError(Exception):
    def __init__(self, action: int, possible_actions: int) -> None:
        super().__init__(
            f"Received action {action}. Action must range between 0 and {possible_actions}"
        )
