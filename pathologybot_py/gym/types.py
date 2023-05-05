from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class EnvState(NamedTuple):
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
    def max_state_value() -> float:
        pass
