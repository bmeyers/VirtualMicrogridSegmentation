#   Actor and Critic DNNs
#   Based on code published by Patrick Emami on his blog "Deep
#   Deterministic Policy Gradients in TensorFlow":
#   https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

import tensorflow as tf
import os

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, tau, gamma,
                 n_layers, size, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.n_layers = n_layers
        self.size = size

        self.critic_lr_placeholder = tf.placeholder(shape=None, dtype=tf.float32)

        # Create the critic network
        self.inputs, self.action, self.out, self.in_training = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self.target_in_training = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            # Define loss and optimization Op
            self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
            self.optimize = tf.train.AdamOptimizer(
                self.critic_lr_placeholder).minimize(self.loss)

            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):

        inputs = tf.placeholder(shape=[None, self.s_dim],
                                dtype=tf.float32,
                                name='observation')
        action = tf.placeholder(shape=[None, self.a_dim],
                                dtype=tf.float32,
                                name='action')
        in_training_mode = tf.placeholder(tf.bool)

        out = tf.layers.flatten(inputs)
        out = tf.keras.layers.Dense(units=self.size, activation=None)(out)
        #out = tf.keras.layers.BatchNormalization()(out,training=in_training_mode)
        out = tf.keras.activations.relu(out)

        t1 = tf.keras.layers.Dense(units=self.size, activation=None)(out)
        t2 = tf.keras.layers.Dense(units=self.size, use_bias=False, activation=None)(action)
        out = tf.keras.layers.Add()([t1, t2])
        #out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
        out = tf.keras.activations.relu(out)
        for i in range(max(self.n_layers - 2, 0)):
            out = tf.keras.layers.Dense(units=self.size, activation=None)(out)
            #out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
            out = tf.keras.activations.relu(out)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003) # Changed from 0.003 values
        out = tf.keras.layers.Dense(units=1, activation=None,
                                    kernel_initializer=w_init)(out)
        #out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
        #out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)

        return inputs, action, out, in_training_mode

    def train(self, inputs, action, predicted_q_value, learning_rate):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.critic_lr_placeholder: learning_rate,
            self.in_training: True
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.in_training: False
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_in_training: False
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.in_training: True
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
