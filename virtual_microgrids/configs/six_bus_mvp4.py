import numpy as np
from scipy.signal import triang
from virtual_microgrids.configs.config_base import ConfigBase

class ConfigSixBusMVP4(ConfigBase):
    """The configurations for the proof of concept (POC) simplest network used in this project.

    The configurations include parameters for the learning algorithm as well as for building and initializing the
    network components. The 6 bus POC is a symmetrical network (actually with 8 buses in this build out), designed
    to show that the two sides can be isolated from each other. To change the values initialized here, change config
    after it is instantiated before using it to build the network.
    """
    def __init__(self, use_baseline, actor):
        self.env_name = 'Six_Bus_MVP4'
        super().__init__(use_baseline, actor, self.env_name)

        self.max_ep_len = 120  # maximum episode length
        self.buffer_size            = 1e6
        self.minibatch_size         = self.max_ep_len * 4
        self.max_episodes           = 1000
        self.reasonable_max_episodes = min(600, self.max_episodes)
        self.max_ep_steps           = self.max_ep_len
        self.randomize_env = True

        # environment generation
        self.tstep = 1. / 60 / 2
        self.net_zero_reward = 1.0
        self.vn_high = 20
        self.vn_low = 0.4
        self.length_km = 0.03
        self.std_type = 'NAYY 4x50 SE'
        n = self.max_ep_len + 1
        self.static_feeds = {
            3: -10 * np.ones(n),
            6: -10 * np.ones(n),
            4: np.random.uniform(9, 11) * np.ones(n),
            7: np.random.uniform(9, 11) * np.ones(n)
        }
        load_types = np.random.choice(['flat', 'sine', 'triangle', 'atan'], size=2)
        for load_type, feed in zip(load_types, [self.static_feeds[4], self.static_feeds[7]]):
            if load_type == 'flat':
                pass
            if load_type == 'sine':
                a = np.random.uniform(-1, 1)
                scale = np.random.uniform(0.5, 2)
                feed += a * np.sin(2 * np.pi * np.arange(n) * scale / n)
            elif load_type == 'triangle':
                a = np.random.uniform(-1, 1)
                roll = np.random.randint(0, n)
                feed += a * 2 * np.roll(triang(n) - 0.5, roll)
            elif load_type == 'atan':
                a = np.random.uniform(-1, 1)
                xs = np.linspace(-5, 5, n)
                feed += a * 2 * np.arctan(xs) / np.pi
        add_step_chance = np.random.uniform(0, 1, size=2)
        for draw, feed in zip(add_step_chance, [self.static_feeds[4], self.static_feeds[7]]):
            if draw <= 0.333:   # probability of introducing a step into the trend
                idxs = np.random.randint(0, n, size=4)
                idxs.sort()
                step = np.zeros(n)
                step[idxs[0]:idxs[2]] += 1
                step[idxs[1]:idxs[3]] += 1
                step /= 2
                a = np.random.uniform(-1, 1)
                feed += a * step
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
        self.reward_epsilon = 0.01
        self.cont_reward_lambda = 0.1

        # parameters for the policy and baseline models
        self.n_layers               = 2
        self.layer_size             = 64

if __name__ == "__main__":
    env = ConfigSixBusMVP3(True, 'DDPG')
