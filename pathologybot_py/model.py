import logging
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue

import numpy as np
import tensorflow as tf

from pathologybot_py.gym.types import Gym, EnvState

NUM_ACTIONS = 4
logger = logging.getLogger(__name__)


@dataclass
class AgentOutputs:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    policies: list[np.ndarray]


class IMPALALoss(tf.keras.losses.Loss):
    def __init__(self, rho_bar=1.0, c_bar=1.0, discount_factor=0.5):
        super().__init__()
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.discount_factor = discount_factor

    def _calculate_rho_s(self, actor_policy, learner_policy) -> tf.Tensor:
        return tf.minimum(tf.constant([self.rho_bar]), learner_policy / actor_policy)
    
    def _calculate_c_t(self, actor_policy, learner_policy) -> tf.Tensor:
        return tf.minimum(tf.constant([self.c_bar]), learner_policy / actor_policy)

    def _compute_vtrace(self, message: AgentOutputs, y_pred: tf.Tensor):
        learner_policy = y_pred[:, :NUM_ACTIONS]
        learner_values = y_pred[:, NUM_ACTIONS]
        actor_policies = np.concatenate(message.policies, axis=0)
        rho_s = self._calculate_rho_s(actor_policies, learner_policy)
        delta_t_V = rho_s * (
            tf.constant(message.rewards[:-1])
            + self.discount_factor * learner_values[1:]
            - learner_values[:-1]
        )
        # TODO: need to figure out the ci product
        c_s = self._calculate_c_s(actor_policies, learner_policy)
        last_vs = learner_values[:-1] + 
        print(f"Shape: {y_pred.shape}")

    def _compute_value_loss(self, vtrace: np.ndarray, y_pred: tf.Tensor) -> tf.Tensor:
        pass

    def call(self, y_true: AgentOutputs, y_pred: tf.Tensor) -> tf.Tensor:
        vtrace = self._compute_vtrace(y_true, y_pred)
        return 0.5 * (y_true - y_pred) ** 2

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.call(y_true, y_pred)


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
                shape=(40, 40, 1),
                batch_size=1 if mode == ModelMode.Actor else None,
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
            ]
            lstm = tf.keras.layers.LSTM(
                256,
                stateful=mode == ModelMode.Actor,
                return_sequences=mode == ModelMode.Learner,
            )
            prior_layer = input
            for layer in layers:
                prior_layer = layer(prior_layer)
            if mode == ModelMode.Learner:
                prior_layer = tf.expand_dims(prior_layer, axis=0)
                prior_layer = lstm(prior_layer)
                prior_layer = tf.squeeze(prior_layer, axis=[0])
            else:
                prior_layer = lstm(tf.expand_dims(prior_layer, axis=1))
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
        policy = model_output[:, :NUM_ACTIONS]
        print(policy)
        action = int(tf.random.categorical(policy, num_samples=1)[0, 0])
        outputs.actions.append(action)
        outputs.rewards.append(env_state.reward)
        outputs.states.append(env_state.state)
        outputs.policies.append(policy)
        if len(outputs.states) >= num_steps_to_send:
            logger.debug("Sending outputs to learner. outputs=%s", outputs)
            queue.put(outputs)
            outputs = AgentOutputs(states=[], actions=[], rewards=[], policies=[])
        return action, outputs

    def _run_actor_loop(
        self,
        gym: Gym,
        outgoing_trajectory_queue: Queue,
        incoming_weights_queue: Queue,
        actor_id: int,
        epochs=200,
        num_steps_to_send=50,
        num_epochs_to_weight_update=10,
    ):
        max_state_output = gym.max_state_value()
        logger.info(
            "Beginning actor loop for actor %d. epochs=%d, num_steps_to_send=%d, num_epochs_to_weight_update=%d",
            actor_id,
            epochs,
            num_steps_to_send,
            num_epochs_to_weight_update,
        )
        for epoch in range(epochs):
            env_state = gym.reset()
            logger.debug("epoch=%d, initial_state=%s", epoch, env_state)
            self._lstm.reset_states()
            if epoch % num_epochs_to_weight_update == 0:
                logger.debug(
                    "Actor %d requesting weight update at epoch %d", actor_id, epoch
                )
                outgoing_trajectory_queue.put(actor_id)
                new_weights = incoming_weights_queue.get()
                logger.debug("Actor %d received new weights to update", actor_id)
                self._model.set_weights(new_weights)
            # initial_lstm_state = self._lstm.get_initial_state()
            outputs = AgentOutputs(states=[], actions=[], rewards=[], policies=[])
            while not env_state.is_final:
                env_state.state /= max_state_output
                env_state.state = np.expand_dims(env_state.state, axis=(0, -1))
                model_output = self._model(env_state.state)
                action, outputs = self._determine_action_and_update_outputs(
                    model_output,
                    env_state,
                    outgoing_trajectory_queue,
                    outputs,
                    num_steps_to_send,
                )
                env_state = gym.step(action)
            env_state.state /= max_state_output
            env_state.state = np.expand_dims(env_state.state, axis=(0, -1))
            self._determine_action_and_update_outputs(
                model_output, env_state, outgoing_trajectory_queue, outputs, 0
            )
        outgoing_trajectory_queue.put(None)

    def _run_learner_loop(
        self,
        incoming_message_queue: "Queue[AgentOutputs | None | int]",
        outgoing_message_queues: list[Queue],
    ):
        num_actors = len(outgoing_message_queues)
        logger.info("Beginning learner loop with %d actors", num_actors)
        nones_received = 0
        while nones_received < num_actors:
            message = incoming_message_queue.get()
            logger.debug("Learner received message: %s", message)
            if message is None:
                nones_received += 1
                logger.debug("Learner was told that an actor has finished")
            elif isinstance(message, int):
                logger.debug("Learner received weight request from actor %d", message)
                outgoing_message_queues[message].put(self._model.get_weights())
            else:
                logger.debug("Training on trajectory: %s", message)
                print([message.shape for message in message.states])
                states = np.concatenate(message.states, axis=0)
                actions = np.asarray(message.actions)
                rewards = np.asarray(message.rewards)
                policies = np.concatenate(message.policies)
                learner_outputs = self._model(states)

                total_loss = self._model.loss(message, learner_outputs)
                self._model.optimizer.optimize(total_loss, self._model.weights)

    def train(
        self,
        gym: Gym,
        trajectory_queue: Queue,
        weights_queues: list[Queue],
        actor_id: int = None,
        epochs=200,
        num_steps_to_send=50,
    ):
        if self._mode == ModelMode.Actor:
            try:
                self._run_actor_loop(
                    gym,
                    trajectory_queue,
                    weights_queues[actor_id],
                    actor_id,
                    epochs,
                    num_steps_to_send,
                )
            except:
                logger.error("Something broke")
                trajectory_queue.put(None)
                raise
        else:
            self._run_learner_loop(trajectory_queue, weights_queues)

    def save_weights_to(self, output):
        self._model.save_weights(output)

    def load_weights_from(self, weights):
        self._model.load_weights(weights)
