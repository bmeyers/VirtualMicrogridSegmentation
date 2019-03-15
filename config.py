import numpy as np
from pandapower.networks import create_synthetic_voltage_control_lv_network as mknet


class ConfigSixBusPOC(object):
    """The configurations for the proof of concept (POC) simplest network used in this project.

    The configurations include parameters for the learning algorithm as well as for building and initializing the
    network components. The 6 bus POC is a symmetrical network (actually with 8 buses in this build out), designed
    to show that the two sides can be isolated from each other. To change the values initialized here, change config
    after it is instantiated before using it to build the network.
    """
    def __init__(self, use_baseline):
        self.env_name = 'Six_Bus_POC'

        # output config
        baseline_str       = 'baseline' if use_baseline else 'no_baseline'
        self.output_path   = "results/{}-{}/".format(self.env_name, baseline_str)
        self.output_path2 = "results/{}-ddpg/".format(self.env_name)
        self.model_output  = self.output_path + "model.weights/"
        self.log_path      = self.output_path + "log.txt"
        self.plot_output   = self.output_path + "scores.png"
        self.record_path   = self.output_path
        self.record_freq   = 5
        self.summary_freq  = 1
        self.summary_freq2 = 200

        # model and training - general
        self.gamma                  = 0.9 # the discount factor

        # model and training config - PG
        self.num_batches            = 500 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 60 # maximum episode length
        self.learning_rate          = 3e-2
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # model and training config - DDPG
        self.tau                    = 0.001
        self.actor_learning_rate    = 1e-3
        self.critic_learning_rate   = 1e-2
        self.buffer_size            = 1e6
        self.minibatch_size         = 64
        self.max_episodes           = self.num_batches * self.batch_size
        self.max_ep_steps           = self.max_ep_len

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
        self.storage_p_min = -50.0
        self.storage_p_max = 50.0

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = None

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class StandardLVNetwork(object):
    """The configurations for using any of the standard low voltage (LV) test networks shipped with pandapower.

    Options in this set up include choices to remove the generation and load elements built in to the test network, and
    the option to remove all sources and sinks of reactive power, q. By adding to the dictionary static_feeds_new you
    can create new loads or static generators on a custom schedule.

    To add controllable resources you can specify the
    locations of new generators, or specify the addition of batteries: either give their locations (by bus number), or
    have them assigned randomly. If percent_battery_buses is non zero (must be in the interval [0, 1]) and
    batteries_on_leaf_nodes_only is False, then percent_battery_buses percent of all the buses will be assigned storage.
    If batteries_on_leaf_nodes_only is True, then percent_battery_buses percent of all the leaf node buses will be
    assigned storage. The initial states of charge (soc) and the capacities can also be changed: these can either be
    floats or lists with length equal to the number of storage elements in the network.
    """
    def __init__(self, env_name, use_baseline):
        self.env_name = env_name

        # output config
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.output_path2 = "results/{}-ddpg/".format(self.env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1
        self.summary_freq2 = 1000

        # model and training - general
        self.gamma                  = 0.9  # the discount factor

        # model and training config - PG
        self.num_batches            = 500  # number of batches trained on
        self.batch_size             = 1000  # number of steps used to compute each policy update
        self.max_ep_len             = 60  # maximum episode length
        self.learning_rate          = 3e-2
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # model and training config - DDPG
        self.tau                    = 0.001
        self.actor_learning_rate    = 1e-3
        self.critic_learning_rate   = 1e-2
        self.buffer_size            = 1e6
        self.minibatch_size         = 64
        self.max_episodes           = 500
        self.max_ep_steps           = self.max_ep_len

        self.remove_q = True
        self.clear_loads_sgen = False
        self.clear_gen = True

        # environment generation
        self.tstep = 1. / 60
        self.net_zero_reward = 1.0
        self.static_feeds_new = None  # Acts how static_feeds does in the 6BusPOC config

        # Fill static_feeds with the loads and static generators that ship with the network
        if self.static_feeds_new is None:
            self.static_feeds = {}
        else:
            self.static_feeds = self.static_feeds_new.copy()
        net = mknet(network_class=env_name)
        if not self.clear_loads_sgen:
            if net.load.shape[0] > 0:
                for idx, row in net.load.iterrows():
                    self.static_feeds[row['bus']] = row['p_kw'] * np.ones(self.max_ep_len)
            if net.sgen.shape[0] > 0:
                for idx, row in net.sgen.iterrows():
                    self.static_feeds[row['bus']] = row['p_kw'] * np.ones(self.max_ep_len)

        self.battery_locations = None  # Specify specific locations, or can pick options for random generation:
        self.percent_battery_buses = 0.5  # How many of the buses should be assigned batteries
        self.batteries_on_leaf_nodes_only = True

        # Action space
        self.gen_p_min = -50.0
        self.gen_p_max = 0.0
        self.storage_p_min = -50.0
        self.storage_p_max = 50.0

        # Generation
        self.gen_locations = [4]
        self.gen_max_p_kw = [20.0]

        self.init_soc = 0.5
        self.energy_capacity = 20.0

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = None

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


def get_config(env_name, baseline=True):
    """Given an environment name and the baseline option, return the configuration."""
    if env_name == 'Six_Bus_POC':
        return ConfigSixBusPOC(baseline)
    if env_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
        return StandardLVNetwork(env_name, baseline)

