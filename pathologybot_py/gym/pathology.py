from enum import Enum

import numpy as np

from pathologybot_py.gym.types import Gym, EnvState


class Difficulty(Enum):
    Easy = 0


class BlockType(Enum):
    Space = 0
    Player = 1
    Exit = 2
    Wall = 3
    Hole = 4
    Full_Block = 5
    L_Block = 6
    LD_Block = 7
    LU_Block = 8
    LR_Block = 9
    LDR_Block = 10
    LDU_Block = 11
    U_Block = 12
    UD_BLOCK = 13
    UR_BLOCK = 14
    UDR_Block = 15
    R_Block = 16
    RD_Block = 17
    D_Block = 18

    @staticmethod
    def max_value():
        return 18


class PathologyGym(Gym):
    def __init__(
        self, starting_difficulty=Difficulty.Easy, difficulty_growth_rate=0.001
    ) -> None:
        self.starting_difficulty = starting_difficulty
        self.difficulty_growth_rate = difficulty_growth_rate

    def reset(self) -> EnvState:
        return EnvState(state=np.random.rand(40, 40, 1), reward=-1, is_final=False)

    def step(self) -> EnvState:
        return EnvState(state=np.random.rand(40, 40, 1), reward=-1, is_final=False)

    def max_state_value() -> float | None:
        return BlockType.max_value()
