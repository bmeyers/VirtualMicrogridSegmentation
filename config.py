class config_sixbus_poc:
    def __init__(self, use_baseline):
        self.env_name = 'Six_Bus_POC'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 60 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = N

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size