import tensorflow as tf

from pathologybot_py.losses import IMPALALoss

import pytest

NUM_ACTIONS = 4


@pytest.fixture
def learner_outputs():
    return tf.convert_to_tensor(
        [
            [-1.0, -7.0, 6.0, 9.0, 9.0],
            [-2.0, 8.0, 7.0, 1.0, 0.0],
            [-7.0, -7.0, -7.0, -7.0, 5.0],
        ]
    )


@pytest.fixture
def actor_outputs():
    return tf.convert_to_tensor(
        [
            [0.0, -6.0, 5.0, 8.0, 8.0],
            [-1.0, 7.0, 6.0, 2.0, 1.0],
            [-6.0, -6.0, -6.0, -6.0, 4.0],
        ]
    )


@pytest.fixture
def learner_values(learner_outputs):
    return learner_outputs[:, NUM_ACTIONS:]


@pytest.fixture
def actor_values(actor_outputs):
    return actor_outputs[:, NUM_ACTIONS:]


@pytest.fixture
def learner_policy_logits(learner_outputs: tf.Tensor):
    return learner_outputs[:, :NUM_ACTIONS]


@pytest.fixture
def actor_policy_logits(actor_outputs: tf.Tensor):
    return actor_outputs[:, :NUM_ACTIONS]


@pytest.fixture
def learner_policy(learner_policy_logits: tf.Tensor):
    # [[4.3244923e-05, 1.0719345e-07, 4.7423821e-02, 9.5253283e-01],
    #  [3.3166798e-05, 7.3054731e-01, 2.6875335e-01, 6.6617294e-04],
    #  [2.5000000e-01, 2.5000000e-01, 2.5000000e-01, 2.5000000e-01]],
    return tf.nn.softmax(learner_policy_logits)


@pytest.fixture
def actor_policy(actor_policy_logits: tf.Tensor):
    # [[3.1945069e-04, 7.9183911e-07, 4.7410689e-02, 9.5226908e-01],
    #  [2.4398121e-04, 7.2729772e-01, 2.6755789e-01, 4.9004937e-03],
    #  [2.5000000e-01, 2.5000000e-01, 2.5000000e-01, 2.5000000e-01]],
    return tf.nn.softmax(actor_policy_logits)


def test_calculate_rho_s(learner_policy: tf.Tensor, actor_policy: tf.Tensor):
    rho_bar = 0.5
    loss = IMPALALoss(NUM_ACTIONS, rho_bar=rho_bar)
    output_rho_s = loss._calculate_rho_s(actor_policy, learner_policy)
    assert tf.math.reduce_all(
        tf.math.equal(
            output_rho_s,
            tf.convert_to_tensor(
                [
                    [0.13537277, 0.13537277, 0.5, 0.5],
                    [0.13593997, 0.5, 0.5, 0.13593997],
                    [0.5, 0.5, 0.5, 0.5],
                ]
            ),
        )
    )


def test_calculate_c_s(learner_policy: tf.Tensor, actor_policy: tf.Tensor):
    c_bar = 0.5
    loss = IMPALALoss(NUM_ACTIONS, c_bar=c_bar)
    output_c_s = loss._calculate_c_s(actor_policy, learner_policy)
    assert tf.math.reduce_all(
        tf.math.equal(
            output_c_s,
            tf.convert_to_tensor(
                [
                    [0.13537277, 0.13537277, 0.5, 0.5],
                    [0.13593997, 0.5, 0.5, 0.13593997],
                    [0.5, 0.5, 0.5, 0.5],
                ]
            ),
        )
    )


def test_compute_value_loss(learner_values):
    mock_vtrace = tf.convert_to_tensor([[1.0], [2.0]])
    loss = IMPALALoss(NUM_ACTIONS)
    value_loss = loss._compute_value_loss(mock_vtrace, learner_values)
    assert tf.reduce_all(tf.math.equal(value_loss, tf.convert_to_tensor(-6.0)))
