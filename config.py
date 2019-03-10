import numpy as np


class ConfigSixBusPOC(object):
    def __init__(self, use_baseline):
        self.env_name = 'Six_Bus_POC'

        # output config
        baseline_str      = 'baseline' if use_baseline else 'no_baseline'
        self.output_path  = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq  = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 60 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # environment generation
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

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = None

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


def get_config(env_name, baseline):
    if env_name == 'Six_Bus_POC':
        return ConfigSixBusPOC(baseline)
