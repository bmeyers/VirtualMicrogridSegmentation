# -*- coding: UTF-8 -*-
"""The base of this code was prepared for a homework by course staff for CS234 at Stanford, Winter 2019."""

import argparse
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from virtual_microgrids.powerflow import NetModel
from virtual_microgrids.utils.general import get_logger, Progbar, export_plot
from virtual_microgrids.configs import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['Six_Bus_POC', 'rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.set_defaults(use_baseline=True)


def build_mlp(mlp_input, output_size, scope, n_layers, size, in_training_mode,
              output_activation=None):
  """
  Build a feed forward network (multi-layer perceptron, or mlp)
  with 'n_layers' hidden layers, each of size 'size' units.
  Use tf.nn.relu nonlinearity between layers.
  Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output layer size
          scope: the scope of the neural network
          n_layers: the number of hidden layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
  Returns:
          The tensor output of the network
  """

  with tf.variable_scope(scope):
    out = tf.layers.flatten(mlp_input)
    for i in range(n_layers):
      out = tf.keras.layers.Dense(units=size, activation=None)(out)
      #out = tf.keras.layers.BatchNormalization()(out, training=in_training_mode)
      out = tf.keras.activations.relu(out)
    w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
    out = tf.layers.Dense(units=output_size, activation=output_activation,
                          kernel_initializer=w_init)(out)

  return out


class PG(object):
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

    self.observation_dim = self.env.observation_dim
    self.action_dim = self.env.action_dim

    self.lr = self.config.learning_rate

    # build model
    self.build()

  def add_placeholders_op(self):
    """
    Add placeholders for observation, action, and advantage:
        self.observation_placeholder, type: tf.float32
        self.action_placeholder, type: depends on the self.discrete
        self.advantage_placeholder, type: tf.float32
    """
    self.observation_placeholder = tf.placeholder(shape=[None, self.observation_dim],
                                                  dtype=tf.float32,
                                                  name='observation')
    self.action_placeholder = tf.placeholder(shape=[None, self.action_dim],
                                             dtype=tf.float32,
                                             name='action')

    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(shape=[None],
                                                dtype=tf.float32,
                                                name='advantage')
    self.in_training_placeholder = tf.placeholder(tf.bool)

  def build_policy_network_op(self, scope = "policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample
    actions from the policy network outputs, and compute the log probabilities
    of the actions taken (for computing the loss later). These operations are
    stored in self.sampled_action and self.logprob.

    Args:
            scope: the scope of the neural network
    """
    action_means = build_mlp(self.observation_placeholder, self.action_dim,
                             scope, self.config.n_layers, self.config.layer_size,
                             self.in_training_placeholder, output_activation=None)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      log_std = tf.get_variable("log_std", [self.action_dim])
    self.sampled_action = action_means + tf.multiply(tf.exp(log_std), tf.random_normal(tf.shape(action_means)))
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=action_means, scale_diag=tf.exp(log_std))
    self.logprob = mvn.log_prob(self.action_placeholder)

  def add_loss_op(self):
    """
    Compute the loss, averaged for a given batch.

    Recall the update for REINFORCE with advantage:
    θ = θ + α ∇_θ log π_θ(a_t|s_t) A_t
    """

    self.loss = - tf.reduce_mean(tf.multiply(self.logprob, self.advantage_placeholder))

  def add_optimizer_op(self):
    """
    Set 'self.train_op' using AdamOptimizer
    """
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_ops):
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      self.train_op = optimizer.minimize(self.loss)

  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope.

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.

    Args:
        scope: the scope of the baseline network

    """

    self.baseline_in_training_placeholder = tf.placeholder(tf.bool)
    self.baseline = tf.squeeze(build_mlp(self.observation_placeholder, 1, scope,
                                         self.config.n_layers, self.config.layer_size,
                                         self.baseline_in_training_placeholder))

    self.baseline_target_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='baseline')

    self.baseline_loss = tf.losses.mean_squared_error(labels=self.baseline_target_placeholder,
                                                      predictions=self.baseline)
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_ops):
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      self.update_baseline_op = optimizer.minimize(self.baseline_loss)

  def build(self):
    """
    Build the model by adding all necessary variables.

    Written by course staff.
    Calling all the operations you already defined above to build the tensorflow graph.
    """

    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optimizer for the main networks
    self.add_optimizer_op()

    # add baseline
    if self.config.use_baseline:
      self.add_baseline_op()

  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    Written by course staff.
    """
    # create tf session
    self.sess = tf.Session()
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
    best_r = 0.0
    best_reward_logical = None

    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0

      soc_track = np.zeros((self.config.max_ep_steps, self.env.net.storage.shape[0]))
      p_track = np.zeros((self.config.max_ep_steps, self.env.net.storage.shape[0]))
      reward_track = np.zeros((self.config.max_ep_steps, 1))

      for step in range(self.config.max_ep_len):
        states.append(state)
        action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None],
                                                               self.in_training_placeholder: False})[0]
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        soc_track[step, :] = self.env.net.storage.soc_percent
        p_track[step, :] = self.env.net.storage.p_kw
        reward_track[step] = reward
        if reward > best_r:
          best_r = reward
          c1 = np.abs(env.net.res_line.p_to_kw - self.env.net.res_line.pl_kw) < self.config.reward_epsilon
          c2 = np.abs(env.net.res_line.p_from_kw - self.env.net.res_line.pl_kw) < self.config.reward_epsilon
          best_reward_logical = np.logical_or(c1.values, c2.values)
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)
          break

      path = {"observation" : np.array(states),
                      "reward" : np.array(rewards),
                      "action" : np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths, episode_rewards, best_r, best_reward_logical, soc_track, p_track, reward_track

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

  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage

    Args:
            returns: all discounted future returns for each step
            observations: observations
    Returns:
            adv: Advantage

    Calculate the advantages, using baseline adjustment if necessary,
    and normalizing the advantages if necessary.
    If neither of these options are True, just return returns.
    """
    adv = returns

    if self.config.use_baseline:
      adv = returns - self.sess.run(self.baseline, feed_dict={self.observation_placeholder: observations,
                                              self.baseline_target_placeholder: returns,
                                              self.baseline_in_training_placeholder: False})

    if self.config.normalize_advantage:
      adv = (adv - np.mean(adv))/np.std(adv)

    return adv

  def update_baseline(self, returns, observations):
    """
    Update the baseline from given returns and observation.

    Args:
            returns: Returns from get_returns
            observations: observations
    """
    self.sess.run(self.update_baseline_op, feed_dict={self.observation_placeholder: observations,
                                                      self.baseline_target_placeholder: returns,
                                                      self.baseline_in_training_placeholder: True})

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
      paths, total_rewards, best_r, best_reward_logical, soc_track, p_track, reward_track = self.sample_path(self.env)
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations,
                    self.action_placeholder : actions,
                    self.advantage_placeholder : advantages,
                    self.in_training_placeholder: True})

      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      best_ep_reward = np.max(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      s1 = "---------------------------------------------------------\n" \
           + "Average reward: {:04.2f} +/- {:04.2f}"
      msg = s1.format(avg_reward, sigma_reward)
      self.logger.info(msg)
      msg4 = "Best episode reward: {}".format(best_ep_reward)
      self.logger.info(msg4)

      msg2 = "Max single reward: " + str(best_r)
      msg3 = "Max reward happened on lines: " + str(best_reward_logical)
      end = "\n--------------------------------------------------------"
      self.logger.info(msg2)
      self.logger.info(msg3 + end)

      fig, ax = plt.subplots(nrows=3, sharex=True)
      xs = np.arange(self.config.max_ep_steps)
      for k_step in range(self.env.net.storage.shape[0]):
        ax[1].plot(xs, soc_track[:, k_step].ravel(), marker='.',
                   label='soc_{}'.format(k_step + 1))
        ax[0].plot(xs, p_track[:, k_step].ravel(), marker='.',
                   label='pset_{}'.format(k_step + 1))
      ax[0].legend()
      ax[1].legend()
      ax[2].stem(xs, reward_track, label='reward')
      ax[2].legend()
      ax[2].set_xlabel('time')
      ax[0].set_ylabel('Power (kW)')
      ax[1].set_ylabel('State of Charge')
      ax[2].set_ylabel('Reward Received')
      ax[0].set_title('Battery Behavior and Rewards')
      plt.tight_layout()
      plt.savefig(self.config.output_path + 'soc_plot_{}.png'.format(t))
      plt.close()

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

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
    #args = parser.parse_args()
    #config = get_config(args.env_name, args.use_baseline)
    config = get_config('Six_Bus_POC', algorithm='PG')
    env = NetModel(config=config)
    # train model
    model = PG(env, config)
    model.run()
