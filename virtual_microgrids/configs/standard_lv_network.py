import numpy as np
from pandapower.networks import create_synthetic_voltage_control_lv_network as mknet
from virtual_microgrids.configs.config_base import ConfigBase


class StandardLVNetwork(ConfigBase):
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
    def __init__(self, env_name, use_baseline, actor):
        self.env_name = env_name
        super().__init__(use_baseline, actor, self.env_name)

        self.reasonable_max_episodes = 1000
        self.max_episodes = 2000

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

        n = self.max_ep_len + 1
        net = mknet(network_class=env_name)
        if not self.clear_loads_sgen:
            if net.load.shape[0] > 0:
                for idx, row in net.load.iterrows():
                    if row['bus'] in self.static_feeds:
                        self.static_feeds[row['bus']].update({0: row['p_kw'] * np.ones(n)})
                    else:
                        self.static_feeds[row['bus']] = {0: row['p_kw'] * np.ones(n)}
            if net.sgen.shape[0] > 0:
                for idx, row in net.sgen.iterrows():
                    if row['bus'] in self.static_feeds:
                        self.static_feeds[row['bus']].update({1: row['p_kw'] * np.ones(n)})
                    else:
                        self.static_feeds[row['bus']] = {1: row['p_kw'] * np.ones(n)}

        self.battery_locations = [19, 23, 24, 22, 8, 10]  # None  # Specify specific locations, or can pick options for random generation:
        self.percent_battery_buses = 0.5  # How many of the buses should be assigned batteries
        self.batteries_on_leaf_nodes_only = True

        # Action space
        self.gen_p_min = -10.0
        self.gen_p_max = 0.0
        self.storage_p_min = -20.0
        self.storage_p_max = 20.0

        # # Generation
        self.gen_locations = None
        # self.gen_locations = [4]
        self.gen_max_p_kw = 20.0

        self.init_soc = 0.5
        self.energy_capacity = 50.0

        # state space
        self.with_soc = False

        # reward function
        self.reward_epsilon = 0.01
        self.cont_reward_lambda = 0.1

        self.moving = True
        self.randomize_env = True  # If this is true, fix the buses where the storage is

        if self.moving:
            for bus, feed in self.static_feeds.items():
                if isinstance(feed, dict):
                    for idx, feed2 in feed.items():
                        a = np.random.uniform(-1, 1)
                        scale = np.random.uniform(0.5, 2)
                        feed2 += a * np.sin(2 * np.pi * np.arange(n) * scale / n)
                else:
                    a = np.random.uniform(-1, 1)
                    scale = np.random.uniform(0.5, 2)
                    feed += a * np.sin(2 * np.pi * np.arange(n) * scale / n)

        self.n_layers = 2
        self.layer_size = 128
