import numpy as np
from pandapower.networks import create_synthetic_voltage_control_lv_network as mknet

class ConfigSixBusPOC(object):
    """The configurations for the proof of concept (POC) simplest network used in this project.

    The configurations include parameters for the learning algorithm as well as for building and initializing the
    network components. The 6 bus POC is a symmetrical network (actually with 8 buses in this build out), designed
    to show that the two sides can be isolated from each other. To change the values initialized here, change config
    after it is instantiated before using it to build the network.
    """
    def __init__(self, use_baseline, actor):
        self.env_name = 'Six_Bus_POC'

        # output config
        baseline_str       = 'baseline' if use_baseline else 'no_baseline'
        self.output_path   = "results/{}-{}-{}/".format(self.env_name, baseline_str, actor)
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
        self.num_batches            = 300 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 60 # maximum episode length
        self.learning_rate          = 3e-2
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # model and training config - DDPG
        self.tau                    = 0.001

        self.reward_epsilon = 0.001

        self.buffer_size            = 1e6
        self.minibatch_size         = self.max_ep_len * 4
        self.max_episodes           = 300
        self.reasonable_max_episodes = min(500, self.max_episodes)
        self.max_ep_steps           = self.max_ep_len

        self.actor_learning_rate_start = 1e-3
        self.actor_learning_rate_end = 1e-6
        self.critic_learning_rate_start = 1e-2
        self.critic_learning_rate_end = 1e-3
        # self.actor_learning_rate_nsteps = self.max_episodes * self.max_ep_steps  # What should this be?

        # environment generation
        self.tstep = 1. / 60
        self.net_zero_reward = 1.0
        self.vn_high = 20
        self.vn_low = 0.4
        self.length_km = 0.03
        self.std_type = 'NAYY 4x50 SE'
        self.static_feeds = {
            3: -10 * np.ones(self.max_ep_len),
            6: -10 * np.ones(self.max_ep_len),
            4: 10 * np.ones(self.max_ep_len),
            7: 10 * np.ones(self.max_ep_len)
        }
        self.battery_locations = [3, 6]
        self.init_soc = 0.5
        self.energy_capacity = 20.0

        # Generation
        self.gen_locations = None

        # Action space
        self.gen_p_min = -50.0
        self.gen_p_max = 0.0
        self.storage_p_min = -10.0
        self.storage_p_max = 10.0

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = None

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size