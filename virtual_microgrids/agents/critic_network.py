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
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

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

        out = tf.layers.flatten(inputs)
        out = tf.layers.dense(out, units=self.size, activation=tf.nn.relu)
        #out = tf.layers.batch_normalization(out)
        #out = tf.nn.relu(out)

        t1 = tf.layers.dense(out, units=self.size)
        t2 = tf.layers.dense(action, units=self.size)

        weights1 = tf.get_default_graph().get_tensor_by_name(
            os.path.split(t1.name)[0] + '/kernel:0')
        weights2 = tf.get_default_graph().get_tensor_by_name(
            os.path.split(t2.name)[0] + '/kernel:0')
        bias = tf.get_default_graph().get_tensor_by_name(
            os.path.split(t1.name)[0] + '/bias:0')
        out = tf.nn.relu(
            tf.matmul(out, weights1) + tf.matmul(action, weights2) + bias)

        for i in range(self.n_layers - 2):
            out = tf.layers.dense(out, units=self.size, activation=tf.nn.relu)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003) # Changed from 0.003 values
        out = tf.layers.dense(out, units=1, kernel_initializer=w_init)

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value, learning_rate):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.critic_lr_placeholder: learning_rate
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
