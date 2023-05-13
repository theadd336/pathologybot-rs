import tensorflow as tf


class IMPALALoss(tf.keras.losses.Loss):
    def __init__(self, num_actions: int, rho_bar=1.0, c_bar=1.0, discount_factor=0.5):
        super().__init__()
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.discount_factor = discount_factor
        self.num_actions = num_actions

    def _calculate_rho_s(self, actor_policy, learner_policy) -> tf.Tensor:
        return tf.minimum(
            tf.constant([self.rho_bar]),
            learner_policy / actor_policy,
        )

    def _calculate_c_s(self, actor_policy, learner_policy) -> tf.Tensor:
        return tf.minimum(
            tf.constant([self.c_bar]),
            learner_policy / actor_policy,
        )

    def _compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        def scanfn(acc, sequence):
            discount_t, c_t, delta_V = sequence
            return delta_V + discount_t * c_t * acc

        learner_policy = tf.nn.softmax(y_pred[:, : self.num_actions])
        learner_values = y_pred[:, self.num_actions :]

        actor_actions = tf.expand_dims(tf.cast(y_true[:, 0], dtype=tf.dtypes.int32), 1)
        actor_rewards = y_true[:, 1:2]
        actor_policies = tf.nn.softmax(y_true[:, 2:])
        rho_s = self._calculate_rho_s(actor_policies, learner_policy)
        actual_actor_policy = tf.expand_dims(
            tf.gather_nd(actor_policies, actor_actions, batch_dims=1), axis=1
        )
        actual_learner_policy = tf.expand_dims(
            tf.gather_nd(learner_policy, actor_actions, batch_dims=1), axis=1
        )
        rho_s = self._calculate_rho_s(actual_actor_policy, actual_learner_policy)
        delta_t_V = rho_s[:-1] * (
            actor_rewards[:-1]
            + self.discount_factor * learner_values[1:]
            - learner_values[:-1]
        )
        c_s = self._calculate_c_s(actual_actor_policy, actual_learner_policy)
        sequences = (
            tf.zeros_like(delta_t_V) + self.discount_factor,
            c_s[:-1],
            delta_t_V,
        )
        initial_values = tf.zeros_like(delta_t_V)
        vs_without_value_addition = tf.scan(
            fn=scanfn,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            reverse=True,
        )
        vs = vs_without_value_addition + learner_values[:-1]

        # TODO: Hyperparameters for the different loss types
        total_loss = self._compute_value_loss(vs, learner_values)
        total_loss += self._compute_policy_loss(
            rho_s, actual_learner_policy, actor_rewards, vs, learner_values
        )
        total_loss += self._compute_entropy_loss(learner_policy)

        return total_loss

    def _compute_value_loss(
        self, vtrace: tf.Tensor, learner_values: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_sum(vtrace - learner_values[:-1])

    def _compute_policy_loss(
        self, rho_s, actual_learner_policy, rewards, vs, learner_values
    ):
        return tf.reduce_sum(
            tf.math.log(actual_learner_policy[:-1])
            * tf.stop_gradient(
                rho_s[:-1]
                * (rewards[:-1] + self.discount_factor * vs[1:] - learner_values[:-1])
            )
        )

    def _compute_entropy_loss(self, learner_policy):
        return -tf.reduce_sum(
            tf.reduce_sum(learner_policy * tf.math.log(learner_policy), axis=-1)
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self._compute_loss(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.call(y_true, y_pred)
