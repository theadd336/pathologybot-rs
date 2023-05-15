import logging
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue

import numpy as np
import tensorflow as tf
import tensorboard as tb

from pathologybot_py.gym.types import Gym, EnvState
from pathologybot_py.losses import IMPALALoss

NUM_ACTIONS = 4
logger = logging.getLogger(__name__)


@dataclass
class AgentOutputs:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    policies: list[np.ndarray]


class ModelSize(Enum):
    Smol = 0
    Lorge = 1


class ModelMode(Enum):
    Learner = 0
    Actor = 1


class ImpalaModel:
    def __init__(self, size: ModelSize, mode=ModelMode.Actor, name=""):
        tf.random.set_seed(1)
        if size == ModelSize.Smol:
            input = tf.keras.Input(
                shape=(40, 40, 1),
                batch_size=1 if mode == ModelMode.Actor else None,
                name=f"{name}-InputLayer",
            )
            layers = [
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=8,
                    strides=(4, 4),
                    activation="relu",
                    name=f"{name}-LargeConv2D",
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=4,
                    strides=(2, 2),
                    activation="relu",
                    name=f"{name}-SmallConv2D",
                ),
                tf.keras.layers.Flatten(name=f"{name}-Flatten"),
                tf.keras.layers.Dense(
                    units=256, activation="relu", name=f"{name}-Dense"
                ),
            ]
            lstm = tf.keras.layers.LSTM(
                256,
                stateful=mode == ModelMode.Actor,
                return_sequences=mode == ModelMode.Learner,
                name=f"{name}-LSTM",
            )
            prior_layer = input
            for layer in layers:
                prior_layer = layer(prior_layer)
            if mode == ModelMode.Learner:
                prior_layer = tf.expand_dims(
                    prior_layer, axis=0, name=f"{name}-LearnerBranchExpandDims"
                )
                prior_layer = lstm(prior_layer)
                prior_layer = tf.squeeze(
                    prior_layer, axis=[0], name=f"{name}-LearnerBranchSqueezeDims"
                )
            else:
                prior_layer = lstm(
                    tf.expand_dims(
                        prior_layer, axis=1, name=f"{name}-ActorBranchExpandDims"
                    )
                )
            outputs = [
                tf.keras.layers.Dense(NUM_ACTIONS, name=f"{name}-PolicyOutput")(
                    prior_layer
                ),
                tf.keras.layers.Dense(1, name=f"{name}-ValueOutput")(prior_layer),
            ]

            model = tf.keras.Model(
                inputs=input,
                outputs=tf.keras.layers.Concatenate(
                    axis=-1, name=f"{name}-ConcateOutputs"
                )(outputs),
            )
        else:
            raise NotImplementedError()
        model.compile(
            optimizer=tf.keras.optimizers.experimental.RMSprop(),
            loss=IMPALALoss(NUM_ACTIONS),
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
        action = int(tf.random.categorical(policy, num_samples=1)[0, 0])
        if action == 4:
            print(policy.numpy())
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
                if new_weights is None:
                    logger.warning(
                        "Actor %d was told that learner crashed. Aborting.", actor_id
                    )
                    outgoing_trajectory_queue.put(None)
                    return
                self._model.set_weights(new_weights)
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
        error_occurred = False
        epoch = 1
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="./logs", histogram_freq=1
        )
        while nones_received < num_actors:
            message = incoming_message_queue.get()
            logger.debug("Learner received message: %s", message)
            if message is None:
                nones_received += 1
                logger.debug("Learner was told that an actor has finished")
            elif error_occurred:
                continue
            elif isinstance(message, int):
                logger.debug("Learner received weight request from actor %d", message)
                outgoing_message_queues[message].put(self._model.get_weights())
            else:
                try:
                    logger.debug("Training on trajectory: %s", message)
                    states = np.concatenate(message.states, axis=0)
                    actions = np.expand_dims(np.asarray(message.actions), axis=1)
                    rewards = np.expand_dims(np.asarray(message.rewards), axis=1)
                    policies = np.concatenate(message.policies, axis=0)
                    vtrace_input = np.hstack([actions, rewards, policies])
                    self._model.fit(
                        x=states,
                        y=vtrace_input,
                        epochs=1,
                        verbose=epoch % 100 == 0,
                        initial_epoch=epoch,
                        callbacks=[tensorboard_callback],
                    )
                    epoch += 1
                except:
                    logger.exception("Fatal error in learner. Aborting training.")
                    for queue in outgoing_message_queues:
                        queue.put(None)
                    error_occurred = True

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
                logger.exception("Something broke")
                trajectory_queue.put(None)
                return
        else:
            self._run_learner_loop(trajectory_queue, weights_queues)

    def evaluate(self, gym: Gym, state: EnvState):
        logger.info("Evaluating round with par: %s", gym._par)
        turns = 0
        max_state_output = gym.max_state_value()
        while not state.is_final:
            turns += 1
            state.state /= max_state_output
            state.state = np.expand_dims(state.state, axis=(0, -1))
            model_output = self._model(state.state)
            policy = model_output[:, :NUM_ACTIONS]
            action = int(tf.random.categorical(policy, num_samples=1)[0, 0])
            state = gym.step(action)

        if state.termination_condition:
            logger.info("Won the stage in %d turns", turns)
        else:
            logger.info("Lost :(")

    def save_weights_to(self, output):
        self._model.save_weights(output)

    def load_weights_from(self, weights):
        self._model.load_weights(weights)
