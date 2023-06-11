from enum import Enum
from typing import Tuple, NamedTuple

import numpy as np

from pathologybot_py.gym.types import Gym, EnvState, InvalidActionError

_STEP_REWARD = -1.0
_LOSS_REWARD = 0.0
_VICTORY_REWARD = 100.0


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


class _Action(Enum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3


class Point(NamedTuple):
    x: int
    y: int


class FinishType(Enum):
    Exit = 0
    OutOfMoves = 1


class _LevelGenerator:
    def __init__(self, test_mode=False) -> None:
        self.test_mode = test_mode
        np.random.seed(0)

    def generate(
        self,
    ) -> Tuple[np.ndarray, int, Point]:
        if self.test_mode:
            state = np.zeros((40, 40)) + BlockType.Wall.value
            for i in range(5):
                for j in range(5):
                    state[i, j] = BlockType.Space.value
            player_start_i = np.random.randint(0, 5)
            player_start_j = np.random.randint(0, 5)
            state[player_start_i, player_start_j] = BlockType.Player.value
            assigned_end = False
            while not assigned_end:
                end_i = np.random.randint(0, 5)
                end_j = np.random.randint(0, 5)
                if end_i == player_start_i and end_j == player_start_j:
                    continue
                assigned_end = True
                state[end_i, end_j] = BlockType.Exit.value
            par = abs(player_start_i - end_i) + abs(player_start_j - end_j)
            return state, par, Point(player_start_i, player_start_j)


class PathologyGym(Gym):
    _NUM_ACTIONS = 4

    def __init__(
        self,
        starting_difficulty=Difficulty.Easy,
        difficulty_growth_rate=0.001,
        test_mode=False,
    ) -> None:
        # Constants for the duration of the gym
        self.starting_difficulty = starting_difficulty
        self.difficulty_growth_rate = difficulty_growth_rate
        self.test_mode = test_mode
        self._level_gen = _LevelGenerator(test_mode=test_mode)

        # Updated via interations with the gym
        self._state: np.ndarray((40, 40))
        self._steps = 0
        self._par = 0
        self._player_pos = Point(0, 0)

    def reset(self) -> EnvState:
        start_state, par, player_pos = self._level_gen.generate()
        self._state = start_state
        self._par = par
        self._steps = 0
        self._player_pos = player_pos
        return EnvState(
            state=start_state, reward=0.0, is_final=False, termination_condition=False
        )

    def step(self, action: int) -> EnvState:
        if action > self._NUM_ACTIONS or action < 0:
            raise InvalidActionError(action, self._NUM_ACTIONS)
        self._steps += 1
        action = _Action(action)
        tentative_player_pos = self._calculate_new_position(action)
        blocked = self._is_blocked(tentative_player_pos)
        finish_type = self._is_final(self._steps, tentative_player_pos)

        if finish_type is None and blocked:
            return EnvState(
                state=self._state,
                reward=_STEP_REWARD,
                is_final=False,
                termination_condition=False,
            )
        elif finish_type is not None and blocked:
            return EnvState(
                state=self._state,
                reward=_LOSS_REWARD,
                is_final=True,
                termination_condition=False,
            )

        self._player_pos = tentative_player_pos
        self._state = self._update_state(
            self._state, tentative_player_pos, self._player_pos, action
        )
        self._player_pos = tentative_player_pos
        if finish_type is not None:
            return EnvState(
                state=self._state,
                reward=_VICTORY_REWARD,
                is_final=True,
                termination_condition=True,
            )
        return EnvState(
            state=self._state,
            reward=_STEP_REWARD,
            is_final=False,
            termination_condition=False,
        )

    def max_state_value(self) -> float | None:
        return BlockType.max_value()

    def _calculate_new_position(self, action: _Action) -> Point:
        match action:
            case _Action.Left:
                return Point(self._player_pos.x - 1, self._player_pos.y)
            case _Action.Right:
                return Point(self._player_pos.x + 1, self._player_pos.y)
            case _Action.Up:
                return Point(self._player_pos.x, self._player_pos.y + 1)
            case _Action.Down:
                return Point(self._player_pos.x, self._player_pos.y - 1)

    def _is_blocked(
        self,
        tentative_player_pos: Point,
    ) -> bool:
        if (
            tentative_player_pos.x < 0 or tentative_player_pos.x >= len(self._state)
        ) or (
            tentative_player_pos.y < 0 or tentative_player_pos.y >= len(self._state[0])
        ):
            return True
        if self._state[tentative_player_pos.x, tentative_player_pos.y] in {
            BlockType.Wall
        }:
            return True
        return False

    def _is_final(self, steps: int, tentative_player_pos: Point) -> FinishType | None:
        if (
            self._state[tentative_player_pos.x, tentative_player_pos.y]
            == BlockType.Exit.value
        ):
            return FinishType.Exit
        if self._par == steps:
            return FinishType.OutOfMoves
        return None

    def _update_state(
        self,
        current_state: np.ndarray,
        new_player_pos: Point,
        old_player_pos: Point,
        action: _Action,
    ) -> np.ndarray:
        # TODO: needs major enhancements
        current_state[new_player_pos.x, new_player_pos.y] = BlockType.Player.value
        current_state[old_player_pos.x, old_player_pos.y] = BlockType.Space.value
        return current_state

    @classmethod
    def num_actions(cls) -> int:
        cls._NUM_ACTIONS
