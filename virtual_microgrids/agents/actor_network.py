#   Actor and Critic DNNs
#   Based on code published by Patrick Emami on his blog "Deep
#   Deterministic Policy Gradients in TensorFlow":
#   https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

import tensorflow as tf

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh, which is individually scaled and
    recentered for each input, to keep each input between p_min and p_max
    for the given device.
    """

    def __init__(self, sess, state_dim, action_dim, tau,
                 n_layers, size, min_p, max_p, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.tau = tau
        self.n_layers = n_layers
        self.size = size
        self.min_p = min_p
        self.max_p = max_p
        self.batch_size = batch_size

        self.actor_lr_placeholder = tf.placeholder(shape=None, dtype=tf.float32)

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.in_training = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_in_training = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

            # Combine the gradients here
            self.unnormalized_actor_gradients = tf.gradients(
                self.scaled_out, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.actor_lr_placeholder). \
                apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):

        inputs = tf.placeholder(shape=[None, self.s_dim],
                                dtype=tf.float32,
                                name='states')
        out = tf.layers.flatten(inputs)
        in_training_mode = tf.placeholder(tf.bool)
        for i in range(self.n_layers):
            out = tf.keras.layers.Dense(units=self.size, activation=None)(out)
            out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
            out = tf.keras.activations.relu(out)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
        out = tf.keras.layers.Dense(units=self.a_dim, activation=None,
                              kernel_initializer=w_init)(out)
        out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
        out = tf.keras.activations.tanh(out)

        centers = (self.min_p + self.max_p) / 2.0
        scales = (self.max_p -self.min_p) / 2.0
        scaled_out = tf.multiply(out, scales) + centers

        return inputs, out, scaled_out, in_training_mode

    def train(self, inputs, a_gradient, learning_rate):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.actor_lr_placeholder: learning_rate,
            self.in_training: True
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.in_training: False
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.target_in_training: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
