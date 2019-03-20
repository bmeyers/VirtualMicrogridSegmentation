import numpy as np
from virtual_microgrids.configs.config_base import ConfigBase

class ConfigSixBusMVP1(ConfigBase):
    """The configurations for the proof of concept (POC) simplest network used in this project.

    The configurations include parameters for the learning algorithm as well as for building and initializing the
    network components. The 6 bus POC is a symmetrical network (actually with 8 buses in this build out), designed
    to show that the two sides can be isolated from each other. To change the values initialized here, change config
    after it is instantiated before using it to build the network.
    """
    def __init__(self, use_baseline, actor):
        self.env_name = 'Six_Bus_MVP1'
        super().__init__(use_baseline, actor, self.env_name)

        # environment generation
        self.tstep = 1. / 60
        self.net_zero_reward = 1.0
        self.vn_high = 20
        self.vn_low = 0.4
        self.length_km = 0.03
        self.std_type = 'NAYY 4x50 SE'
        self.static_feeds = {
            3: -10 * np.ones(self.max_ep_len + 1),
            6: -10.5 * np.ones(self.max_ep_len + 1),
            4: 10.5 * np.ones(self.max_ep_len + 1),
            7: 10 * np.ones(self.max_ep_len + 1)
        }
        self.battery_locations = [3, 6]
        self.init_soc = 0.5
        self.energy_capacity = 21.0  # changed from 20 to see if endpoint problem

        # Generation
        self.gen_locations = None

        # Action space
        self.gen_p_min = -50.0
        self.gen_p_max = 0.0
        self.storage_p_min = -5.0
        self.storage_p_max = 5.0

        # state space
        self.with_soc = False

        # reward function
        self.reward_epsilon = 0.001
        self.cont_reward_lambda = 0.1
