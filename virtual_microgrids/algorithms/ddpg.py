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
import matplotlib.pyplot as plt

sys.path.append('..')
from virtual_microgrids.powerflow import NetModel
from virtual_microgrids.utils.general import get_logger, Progbar, export_plot
from virtual_microgrids.configs import get_config
from virtual_microgrids.utils import ReplayBuffer, LinearSchedule, OrnsteinUhlenbeckActionNoise
from virtual_microgrids.agents import ActorNetwork, CriticNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['Six_Bus_POC', 'rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1'])


class DDPG(object):
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

        # self.actor_lr = self.config.actor_learning_rate_start
        # self.critic_lr = self.config.critic_learning_rate_start
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.batch_size = self.config.minibatch_size

        # self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
        self.actor_noise = lambda noise_level: np.random.normal(0, noise_level, size=self.action_dim) # changed from 0.2

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
        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.tau, self.config.n_layers,
                                  self.config.layer_size, self.min_p, self.max_p,
                                  self.config.minibatch_size)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.tau, self.gamma,
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
        # new DDPG placeholders
        self.max_q_placeholder = tf.placeholder(tf.float32, shape=(), name='max_q')

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)
        # new DDPG summary
        tf.summary.scalar("Max_Q_Value", self.max_q_placeholder)

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
        self.avg_max_q = 0.

    def update_averages(self, rewards, scores_eval, avg_max_q):
        """
        Update the averages. Written by course staff.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
        self.avg_max_q = np.mean(avg_max_q)

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
            self.max_q_placeholder: self.avg_max_q
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

    def train(self):
        """
        Performs training.
        """

        actor_lr_schedule = LinearSchedule(self.config.actor_learning_rate_start, self.config.actor_learning_rate_end,
                                           self.config.reasonable_max_episodes*self.config.max_ep_steps)
        critic_lr_schedule = LinearSchedule(self.config.critic_learning_rate_start, self.config.critic_learning_rate_end,
                                            self.config.reasonable_max_episodes*self.config.max_ep_steps)
        noise_schedule = LogSchedule(0.5, 0.01, self.config.reasonable_max_episodes*self.config.max_ep_steps)
        # works with 0.5, 0.01 Linear

        self.actor.update_target_network()
        self.critic.update_target_network()
        replay_buffer = ReplayBuffer(self.config.buffer_size)
        total_rewards = []
        scores_eval = []
        ave_max_q = []

        for i in range(self.config.max_episodes):
            s = self.env.reset()
            ep_reward = 0
            ep_ave_max_q = 0
            best_ep_reward = 0

            best_r = 0.0
            best_reward_logical = None

            soc_track = np.zeros((self.config.max_ep_steps, self.env.net.storage.shape[0]))
            achieved_1 = np.zeros((self.config.max_ep_steps, 1))
            achieved_2 = np.zeros((self.config.max_ep_steps, 1))
            achieved_3 = np.zeros((self.config.max_ep_steps, 1))

            for j in range(self.config.max_ep_steps):
                a = self.actor.predict(s[None, :]) + self.actor_noise(noise_schedule.epsilon)
                s2, r, done, info = self.env.step(a[0])
                replay_buffer.add(np.reshape(s, (self.state_dim)),
                                  np.reshape(a, (self.action_dim)),
                                  r, done,
                                  np.reshape(s2, (self.state_dim)))
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > self.config.minibatch_size:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(self.config.minibatch_size)
                    # Calc targets
                    target_q = self.critic.predict_target(
                        s2_batch, self.actor.predict_target(s2_batch)
                    )
                    y_i = np.array(r_batch)
                    y_i[~t_batch] = (r_batch +
                                     self.gamma * target_q.squeeze())[~t_batch]
                    # Update critic given targets
                    predicted_q_val, _ = self.critic.train(s_batch, a_batch, y_i[:, None], critic_lr_schedule.epsilon)
                    ep_ave_max_q += np.max(predicted_q_val)
                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0], actor_lr_schedule.epsilon)
                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()
                    actor_lr_schedule.update(i*self.config.max_ep_steps + j)
                    critic_lr_schedule.update(i * self.config.max_ep_steps + j)
                    noise_schedule.update(i * self.config.max_ep_steps + j)
                # Housekeeping
                if r > best_r:
                    best_r = r
                    c1 = np.abs(self.env.net.res_line.p_to_kw - self.env.net.res_line.pl_kw) < self.config.reward_epsilon
                    c2 = np.abs(self.env.net.res_line.p_from_kw - self.env.net.res_line.pl_kw) < self.config.reward_epsilon
                    best_reward_logical = np.logical_or(c1.values, c2.values)

                soc_track[j, :] = self.env.net.storage.soc_percent
                if r == 1:
                    achieved_1[j] = 1.0
                elif r == 2:
                    achieved_2[j] = 1.0
                elif r == 3:
                    achieved_3[j] = 1.0

                s = s2
                ep_reward += r
                if done:
                    if ep_reward > best_ep_reward:
                        best_ep_reward = ep_reward
                    total_rewards.append(ep_reward)
                    ep_ave_max_q /= j
                    ave_max_q.append(ep_ave_max_q)
                    break

            # tf stuff
            if (i % self.config.summary_freq2 == 0):
                scores_eval.extend(total_rewards)
                self.update_averages(np.array(total_rewards), np.array(scores_eval), np.array(ave_max_q))
                self.record_summary(i)

                # compute reward statistics for this batch and log
                avg_reward = np.mean(total_rewards)
                sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
                avg_q = np.mean(ave_max_q)
                s1 = "Average reward: {:04.2f} +/- {:04.2f}    Average Max Q: {:.2f}"
                msg = s1.format(avg_reward, sigma_reward, avg_q)
                self.logger.info(msg)

                msg2 = "The max episode reward achieved as: "+str(best_r)
                msg3 = "The rewards happened on which lines: "+str(best_reward_logical)
                self.logger.info(msg2)
                self.logger.info(msg3)

                msg4 = "The best episode reward was {}".format(best_ep_reward)
                self.logger.info(msg4)

                plt.figure()
                plt.plot(np.arange(0, self.config.max_ep_steps), achieved_1, '*')
                plt.plot(np.arange(0, self.config.max_ep_steps), achieved_2, '*')
                plt.plot(np.arange(0, self.config.max_ep_steps), achieved_3, '*')
                for k_step in range(self.env.net.storage.shape[0]):
                    plt.plot(np.arange(0, self.config.max_ep_steps), soc_track[:, k_step], '.')
                plt.legend(labels=['Achieved r = 1', 'Achieved r = 2', 'Achieved r = 3'])
                # plt.xlabel('Episode steps', fontname='Courier')
                # plt.ylabel('SOC Percent', fontname='Courier')
                # plt.title('Average reward: '+str(avg_reward)+' +/- '+str(sigma_reward), loc='center', fontname='Courier')
                plt.savefig(self.config.output_path + 'soc_plot.png')  # , bbox_inches='tight')
                plt.close()

                total_rewards = []
                ave_max_q = []
                best_ep_reward = 0

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

    config = get_config('Six_Bus_POC')
    env = NetModel(config=config)
    # train model
    model = DPG(env, config)
    model.run()
