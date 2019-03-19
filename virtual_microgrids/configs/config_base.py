import numpy as np
from datetime import datetime as dt

class ConfigBase(object):
    """A base class for configurations"""
    def __init__(self, use_baseline, actor, env_name):

        # output config
        now  = dt.now()
        now =  ''.join('_'.join(str(now).split(' ')).split(':'))
        baseline_str       = 'baseline' if use_baseline else 'no_baseline'
        self.output_path   = "results/{}-{}-{}_{}/".format(env_name, baseline_str, actor, now)
        self.model_output  = self.output_path + "model.weights/"
        self.log_path      = self.output_path + "log.txt"
        self.plot_output   = self.output_path + "scores.png"
        self.record_path   = self.output_path
        self.record_freq   = 5
        self.summary_freq  = 1
        self.summary_freq2 = 20

        # model and training - general
        self.gamma                  = 0.9 # the discount factor

        # model and training config - PG
        self.num_batches            = 150 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 60 # maximum episode length
        self.learning_rate          = 3e-2
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # model and training config - DDPG
        self.tau                    = 0.001

        self.buffer_size            = 1e6
        self.minibatch_size         = self.max_ep_len * 4
        self.max_episodes           = 1000
        self.reasonable_max_episodes = min(600, self.max_episodes)
        self.max_ep_steps           = self.max_ep_len

        self.actor_learning_rate_start = 1e-3
        self.actor_learning_rate_end = 1e-6
        self.critic_learning_rate_start = 1e-2
        self.critic_learning_rate_end = 1e-3
        # self.actor_learning_rate_nsteps = self.max_episodes * self.max_ep_steps  # What should this be?

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = None

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size