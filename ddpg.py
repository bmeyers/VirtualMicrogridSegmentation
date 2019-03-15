# -*- coding: UTF-8 -*-
"""The base of this code was prepared for a homework by course staff for CS234 at Stanford, Winter 2019. We have since
altered it to implement DDPG rather than traditional PG. Also inspired by code published by Patrick Emami on his blog
"Deep Deterministic Policy Gradients in TensorFlow": https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
"""

import os
import argparse
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import scipy.signal
import os
import time
import inspect
from collections import deque

from pp_network import NetModel
from utils.general import get_logger, Progbar, export_plot
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['Six_Bus_POC', 'rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1'])
# parser.add_argument('--baseline', dest='use_baseline', action='store_true')
# parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
# parser.set_defaults(use_baseline=True)


class ReplayBuffer(object):
  """
  A data structure to hold the replay buffer. Based on this blog post:
  https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
  """
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.count = 0
    self.buffer = deque()

  def add(self, s, a, r, t, s2):
    experience = (s, a, r, t, s2)
    if self.count < self.buffer_size:
      self.buffer.append(experience)
      self.count += 1
    else:
      self.buffer.popleft()
      self.buffer.append(experience)

  def size(self):
    return self.count

  def sample_batch(self, batch_size):
    '''
    batch_size specifies the number of experiences to add
    to the batch. If the replay buffer has less than batch_size
    elements, simply return all of the elements within the buffer.
    Generally, you'll want to wait until the buffer has at least
    batch_size elements before beginning to sample from it.
    '''
    batch = []

    if self.count < batch_size:
      batch = random.sample(self.buffer, self.count)
    else:
      batch = random.sample(self.buffer, batch_size)

    s_batch = np.array([_[0] for _ in batch])
    a_batch = np.array([_[1] for _ in batch])
    r_batch = np.array([_[2] for _ in batch])
    t_batch = np.array([_[3] for _ in batch])
    s2_batch = np.array([_[4] for _ in batch])

    return s_batch, a_batch, r_batch, t_batch, s2_batch

  def clear(self):
    self.buffer.clear()
    self.count = 0

# ===========================
#   Actor and Critic DNNs
#   Created working with code published by Patrick Emami on his blog "Deep
#   Deterministic Policy Gradients in TensorFlow":
#   https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau,
                 n_layers, size, min_p, max_p, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.n_layers = n_layers
        self.size = size
        self.min_p = min_p
        self.max_p = max_p
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim],
                                                  dtype=tf.float32,
                                                  name='states')
        out = tf.layers.flatten(inputs)
        for i in range(self.n_layers):
          out = tf.layers.dense(out, units=self.size)
          out = tf.layers.batch_normalization(out)
          out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=self.a_dim, activation=tf.nn.tanh)

        centers = (self.min_p + self.max_p) / 2.0
        scales = (self.max_p -self.min_p) / 2.0
        scaled_out = tf.multiply(out, scales) + centers

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma,
                 n_layers, size, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.n_layers = n_layers
        self.size = size

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
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

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
        out = tf.layers.dense(out, units=self.size, activation=None)
        out = tf.layers.batch_normalization(out)
        out = tf.nn.relu(out)

        t1 = tf.layers.dense(out, units=self.size)
        t2 = tf.layers.dense(action, units=self.size)

        out = tf.nn.relu(
          tf.matmul(out, t1.W) + tf.matmul(action, t2.W) + t2.b)

        for i in range(self.n_layers - 2):
          out = tf.layers.dense(out, units=size, activation=tf.nn.relu)

        out = tf.layers.dense(out, units=1)

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
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


class OrnsteinUhlenbeckActionNoise(object):
  """
  Implementation of an Ornstein–Uhlenbeck process for exploration. Based on:
  https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
  https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
  """
  def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

  def __call__(self):
    x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
         self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu,
                                                            self.sigma)

class DPG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, config, logger=None):
    """
    Initialize Policy Gradient Class

    Args:
            env: an OpenAI Gym environment
            config: class with hyperparameters
            logger: logger instance from the logging module

    Written by course staff.
    """
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)

    # store hyperparameters
    self.config = config
    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)
    self.env = env

    self.state_dim = self.env.observation_dim
    self.action_dim = self.env.action_dim

    self.actor_lr = self.config.actor_learning_rate
    self.critic_lr = self.config.critic_learning_rate
    self.gamma = self.config.gamma
    self.tau = self.config.tau
    self.batch_size = self.config.minibatch_size

    self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))

    # action space limits
    min_p = []
    max_p = []
    if len(env.net.gen)>0:
      min_p.append(env.net.gen.min_p_kw)
      max_p.append(env.net.gen.max_p_kw)
    if len(env.net.storage)>0:
      min_p.append(env.net.storage.min_p_kw)
      max_p.append(env.net.storage.max_p_kw)
    self.min_p = np.array(min_p)
    self.max_p = np.array(max_p)

    # build model
    self.actor = None
    self.critic = None

  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    Written by course staff.
    """
    # create tf session
    self.sess = tf.Session()
    # Initialize networks
    self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim,
                              self.actor_lr, self.tau, self.config.n_layers,
                              self.config.layer_size, self.min_p, self.max_p,
                              self.config.minibatch_size)
    self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim,
                                self.critic_lr, self.tau, self.gamma,
                                self.config.n_layers, self.config.layer_size,
                                self.actor.get_num_trainable_vars())
    # tensorboard stuff
    self.add_summary()
    # initialize all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def add_summary(self):
    """
    Tensorboard stuff. Written by course staff.
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph)

  def init_averages(self):
    """
    Defines extra attributes for tensorboard. Written by course staff.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages. Written by course staff.

    Args:
        rewards: deque
        scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

  def record_summary(self, t):
    """
    Add summary to tensorboard. Written by course staff.
    """

    fd = {
      self.avg_reward_placeholder: self.avg_reward,
      self.max_reward_placeholder: self.max_reward,
      self.std_reward_placeholder: self.std_reward,
      self.eval_reward_placeholder: self.eval_reward,
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

  def sample_path(self, env, num_episodes = None):
    """
    Sample paths (trajectories) from the environment.

    Args:
        num_episodes: the number of episodes to be sampled
            if none, sample one batch (size indicated by config file)
        env: open AI Gym envinronment

    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    Written by course staff.
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0

    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0

      for step in range(self.config.max_ep_len):
        states.append(state)
        action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})[0]
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      path = {"observation" : np.array(states),
                      "reward" : np.array(rewards),
                      "action" : np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths, episode_rewards

  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep

    Args:
            paths: recorded sample paths.  See sample_path() for details.

    Return:
            returns: return G_t for each timestep

    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):

       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

    where T is the last timestep of the episode.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]

      dim_rewards = np.shape(np.ravel(rewards))[0] # Each path has a different length
      returns = np.zeros((dim_rewards,))
      for i in range(dim_rewards):
        for j in range(dim_rewards-i):
          returns[i] += rewards[i+j]*np.power(self.config.gamma, j)  # Implement the sum in the G_t formula

      all_returns.append(returns)
    returns = np.concatenate(all_returns)

    return returns

  # def calculate_advantage(self, returns, observations):
  #   """
  #   Calculate the advantage
  #
  #   Args:
  #           returns: all discounted future returns for each step
  #           observations: observations
  #   Returns:
  #           adv: Advantage
  #
  #   Calculate the advantages, using baseline adjustment if necessary,
  #   and normalizing the advantages if necessary.
  #   If neither of these options are True, just return returns.
  #   """
  #   adv = returns
  #
  #   if self.config.use_baseline:
  #     adv = returns - self.sess.run(self.baseline, feed_dict={self.observation_placeholder: observations,
  #                                             self.baseline_target_placeholder: returns})
  #
  #   if self.config.normalize_advantage:
  #     adv = (adv - np.mean(adv))/np.std(adv)
  #
  #   return adv

  # def update_baseline(self, returns, observations):
  #   """
  #   Update the baseline from given returns and observation.
  #
  #   Args:
  #           returns: Returns from get_returns
  #           observations: observations
  #   """
  #   self.sess.run(self.update_baseline_op, feed_dict={self.observation_placeholder: observations,
  #                                                     self.baseline_target_placeholder: returns})

  def update_critic(self, returns, observations, actions):

    self.sess.run(self.update_critic_op, feed_dict={self.observation_placeholder: observations,
                                                    self.returns_placeholder: returns,
                                                    self.action_placeholder: actions})

  def train(self):
    """
    Performs training. Written by course staff.
    """
    last_eval = 0
    last_record = 0
    scores_eval = []

    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time

    for t in range(self.config.num_batches):

      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env)
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      # advantages = self.calculate_advantage(returns, observations)

      # run training operations
      # if self.config.use_baseline:
      #   self.update_baseline(returns, observations)
      self.update_critic(returns, observations, actions)
      #### TODO is this feeding it the right actions and observations? Should it not be the next observations for the
      # critic?

      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations,
                    self.action_placeholder : actions}) #  ,self.advantage_placeholder : advantages})

      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)

  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes. Written by course staff.
    Not used right now, all evaluation statistics are computed during training
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward

  def run(self):
    """
    Apply procedures of training for a PG. Written by course staff.
    """
    # initialize
    self.initialize()
    # model
    self.train()

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)  #, args.use_baseline)
    env = NetModel(config=config)
    # train model
    model = DPG(env, config)
    model.run()
