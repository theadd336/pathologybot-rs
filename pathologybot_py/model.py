from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue

import numpy as np
import tensorflow as tf

from pathologybot_py.gym.types import Gym, EnvState

NUM_ACTIONS = 4


@dataclass
class AgentOutputs:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    policies: list[np.ndarray]


class IMPALALoss(tf.keras.losses.Loss):
    def __init__(self, rho_bar=1.0, c_bar=1.0):
        super().__init__()
        self.rho_bar = rho_bar
        self.c_bar = c_bar

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        print(y_pred)
        # raise ValueError(f"{y_true, y_pred}")
        return 0.5 * (y_true - y_pred) ** 2


class ModelSize(Enum):
    Smol = 0
    Lorge = 1


class ModelMode(Enum):
    Learner = 0
    Actor = 1


class ImpalaModel:
    def __init__(self, size: ModelSize, mode=ModelMode.Actor, name=""):
        if size == ModelSize.Smol:
            input = tf.keras.Input(
                shape=(40, 40, 1), batch_size=1 if mode == ModelMode.Actor else None
            )
            layers = [
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=8,
                    strides=(4, 4),
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=4, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Reshape(target_shape=(1, 256)),
            ]
            lstm = tf.keras.layers.LSTM(256, stateful=mode == ModelMode.Actor)
            prior_layer = input
            for layer in layers:
                prior_layer = layer(prior_layer)
            prior_layer = lstm(prior_layer)
            outputs = [
                tf.keras.layers.Dense(NUM_ACTIONS)(prior_layer),
                tf.keras.layers.Dense(1)(prior_layer),
            ]

            model = tf.keras.Model(
                inputs=input, outputs=tf.keras.layers.Concatenate(axis=-1)(outputs)
            )
        else:
            raise NotImplementedError()
        model.compile(
            optimizer=tf.keras.optimizers.experimental.RMSprop(), loss=IMPALALoss()
        )
        self._model = model
        self._lstm = lstm
        self._mode = mode

    def _determine_action_and_update_outputs(
        self,
        model_output: np.ndarray,
        env_state: EnvState,
        queue: Queue,
        outputs: AgentOutputs,
        num_steps_to_send: int,
    ) -> AgentOutputs:
        policy = model_output[0][:NUM_ACTIONS]
        action = tf.random.categorical(policy, num_samples=1)
        outputs.actions.append(action)
        outputs.rewards.append(env_state.reward)
        outputs.states.append(env_state.state)
        outputs.policies.append(policy)
        if len(outputs.states) >= num_steps_to_send:
            queue.put(outputs)
            outputs = AgentOutputs()
        return action, outputs

    def _run_actor_loop(self, gym: Gym, queue: Queue, epochs=200, num_steps_to_send=50):
        max_state_output = gym.max_state_value()
        for _ in range(epochs):
            env_state = gym.reset()
            self._lstm.reset_states()
            # initial_lstm_state = self._lstm.get_initial_state()
            outputs = AgentOutputs()
            while not env_state.is_final:
                model_output = self._model(env_state.state / max_state_output)
                action, outputs = self._determine_action_and_update_outputs(
                    model_output, env_state, queue, outputs, num_steps_to_send
                )
                env_state = gym.step(action)
            self._determine_action_and_update_outputs(
                model_output, env_state, queue, outputs, 0
            )
        queue.put(None)

    def train(self, gym: Gym, queue: Queue, epochs=200, num_steps_to_send=50):
        if self._mode == ModelMode.Actor:
            self._run_actor_loop(gym, queue, epochs, num_steps_to_send)
