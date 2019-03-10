import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.networks import \
    create_synthetic_voltage_control_lv_network as mknet

from config import get_config
from network_generation import get_net


class NetModel(object):
    """Building and interacting with a network model to simulate power flow.

    In this class we model all of the network component including loads,
    generators, batteries, lines, buses, and transformers. The state of each is
    tracked in a pandapower network object.
    """

    def __init__(self, net_given=None, network_name='rural_1',
                 zero_out_gen_shunt_storage=False, tstep=1. / 60,
                 net_zero_reward=1.0, env_name=None, baseline=True,
                 config=None):
        """Initialize attributes of the object and zero out certain components
        in the standard test network."""

        if config is not None:
            self.config = config
            self.net = get_net(self.config)
        elif env_name is not None:
            self.config = get_config(env_name, baseline)
            self.net = get_net(self.config)
        elif net_given is not None:
            self.net = net_given
            self.network_name = 'custom_network'
            self.config = None
        else:
            self.net = mknet(network_class=network_name)
            self.network_name = network_name
            self.config = None

        self.reward_val = 0.0

        self.tstep = tstep
        self.net_zero_reward = net_zero_reward
        self.initial_net = pp.copy.deepcopy(self.net)
        self.time = 0
        self.n_load = len(self.net.load)
        self.n_sgen = len(self.net.sgen)
        self.n_gen = len(self.net.gen)
        self.n_storage = len(self.net.storage)
        self.observation_dim = self.n_load + self.n_sgen + self.n_gen + 2 * self.n_storage
        self.action_dim = self.n_gen + self.n_storage

    def reset(self):
        """Reset the network and reward values back to how they were initialized."""
        self.net = pp.copy.deepcopy(self.initial_net)
        self.reward_val = 0.0
        self.time = 0
        self.run_powerflow()
        state = self.get_state()
        return state

    def step(self, p_set):
        """Update the simulation by one step

        :param p_set: 1D numpy array of floats, the action for the agent
        :return:
        """
        # Increment the time
        self.time += 1
        # Update non-controllable resources from their predefined data feeds
        new_loads = pd.Series(data=None, index=self.net.load.bus)
        new_sgens = pd.Series(data=None, index=self.net.sgen.bus)
        for bus, feed in self.config.static_feeds.items():
            p_new = feed[self.time]
            if p_new > 0:
                new_loads[bus] = p_new
            else:
                new_sgens[bus] = p_new
        self.update_loads(new_p=new_loads.values)
        self.update_static_generation(new_p=new_sgens.values)
        # Update controllable resources
        new_gens = p_set[:self.n_gen]
        new_storage = p_set[self.n_gen:]
        self.update_generation(new_p=new_gens)
        self.update_batteries(new_p=new_storage)
        # Run power flow
        self.run_powerflow()
        # Collect items to return
        state = self.get_state()
        reward = self.calculate_reward()
        done = self.time >= self.config.max_ep_len
        info = ''
        return state, reward, done, info

    def get_state(self):
        """Get the current state of the game

        The state is given by the power supplied or consumed by all devices
        on the network, plus the state of charge (SoC) of the batteries. This
        method defines a "global ordering" for this vector:
            - Non-controllable loads (power, kW)
            - Non-controllable generators (power, kW)
            - Controllable generators (power, kW)
            - Controllable batteries (power, kW)
            - SoC for batteries (soc, no units)

        We are not currently considering reactive power (Q) as part of the
        problem.

        :return: A 1D numpy array containing the current state
        """
        p_load = self.net.res_load.p_kw
        p_sgen = self.net.res_sgen.p_kw
        p_gen = self.net.res_gen.p_kw
        p_storage = self.net.res_storage.p_kw
        soc_storage = self.net.storage.soc_percent
        state = np.concatenate([p_load, p_sgen, p_gen, p_storage, soc_storage])
        return state

    def add_sgen(self, bus_number, init_real_power, init_react_power=0.0):
        """Change the network by adding a static generator.

        Parameters
        ----------
        bus_number: int
            The bus at which the static generator should be added
        init_real_power: float
            The real power generation of the static generator for initialization.
        init_react_power: float
            The reactive power generation of the static generator for initialization.

        Attributes
        ----------
        net: object
            The network object is updated
        """
        pp.create_sgen(self.net, bus_number, init_real_power, init_react_power)

    def add_generation(self, bus_number, init_real_power, set_limits=False,
                       min_p_kw=0, max_p_kw=0, min_q_kvar=0,
                       max_q_kvar=0):
        """Change the network by adding a traditional generator.

        Parameters
        ----------
        bus_number: int
            The bus at which the generator should be added
        init_real_power: float
            The real power generation of the generator for initialization.
        set_limits: bool
            Whether or not the initialization includes limits on the real and reactive power flows.
        min_p_kw, max_p_kw, min_q_kvar, max_q_kvar: float
            Power limits on the generator.

        Attributes
        ----------
        net: object
            The network object is updated
        """
        if set_limits:
            pp.create_gen(self.net, bus_number, init_real_power,
                          min_p_kw=min_p_kw, max_p_kw=max_p_kw,
                          min_q_kvar=min_q_kvar, max_q_kvar=max_q_kvar)
        else:
            pp.create_gen(self.net, bus_number, init_real_power)

    def update_loads(self, new_p=None, new_q=None):
        """Update the loads in the network.

        This method assumes that the orders match, i.e. the order the buses in
        self.net.load.bus matches where the loads in new_p and new_q should be
        applied based on their indexing.

        Parameters
        ----------
        new_p, new_q: array_like
            New values for the real and reactive load powers, shape (number of load buses, 1).

        Attributes
        ----------
        self.net.load: object
            The load values in the network object are updated.
        """
        if new_p is not None:
            self.net.load.p_kw = new_p
        if new_q is not None:
            self.net.load.q_kvar = new_q

    def update_static_generation(self, new_p=None, new_q=None):
        """Update the static generation in the network.

        This method assumes that the orders match, i.e. the order the buses in
        self.net.sgen.bus matches where the generation values in new_sgen_p and
        new_sgen_q should be applied based on their indexing.

        Parameters
        ----------
        new_sgen_p, new_sgen_q: array_like
            New values for the real and reactive static generation, shape
            (number of static generators, 1).

        Attributes
        ----------
        self.net.sgen: object
            The static generation values in the network object are updated.
        """
        if new_p is not None:
            self.net.sgen.p_kw = new_p
        if new_q is not None:
            self.net.sgen.q_kvar = new_q

    def update_generation(self, new_p=None, new_q=None):
        """Update the traditional (not static) generation in the network.

        This method assumes that the orders match, i.e. the order the buses in
        self.net.gen.bus matches where the generation values in new_gen_p
        should be applied based on their indexing.

        Parameters
        ----------
        new_gen_p: array_like
            New values for the real and reactive generation, shape (number of
            traditional generators, 1).

        Attributes
        ----------
        self.net.gen: object
            The traditional generation values in the network object are updated.
        """
        if new_p is not None:
            self.net.gen.p_kw = new_p
        if new_q is not None:
            self.net.gen.q_kvar = new_q

    def update_batteries(self, new_p):
        """Update the batteries / storage units in the network.

        This method assumes that the orders match, i.e. the order the buses in
        self.net.gen.bus matches where the generation values in new_gen_p
        should be applied based on their indexing.

        Parameters
        ----------
        battery_powers: array_like
            The power flow into / out of each battery, shape (number of traditional generators, 1).

        Attributes
        ----------
        self.net.storage: object
            The storage values in the network object are updated.
        """
        soc = self.net.storage.soc_percent
        cap = self.net.storage.max_e_kwh
        eff = self.net.storage.eff
        pmin = self.net.storage.min_p_kw
        pmin_soc = -1 * soc * cap * eff / self.tstep
        pmin = np.max([pmin, pmin_soc], axis=0)
        pmax = self.net.storage.max_p_kw
        pmax_soc = (1. - soc) * cap / (eff * self.tstep)
        pmax = np.min([pmax, pmax_soc], axis=0)
        ps = np.clip(new_p, pmin, pmax)
        self.net.storage.p_kw = ps
        soc_next = soc + ps * self.tstep * eff / cap
        msk = ps < 0
        soc_next[msk] = (soc + ps * self.tstep / (eff * cap))[msk]
        self.net.storage.soc_percent = soc_next

    def run_powerflow(self):
        """Evaluate the power flow. Results are stored in the results matrices
        of the net object, e.g. self.net.res_bus.

        Attributes
        ----------
        self.net: object
            The network matrices are updated to reflect the results.
            Specifically: self.net.res_bus, self.net.res_line, self.net.res_gen,
            self.net.res_sgen, self.net.res_trafo, self.net.res_storage.
        """
        try:
            pp.runpp(self.net, enforce_q_lims=True,
                     calculate_voltage_angles=False,
                     voltage_depend_loads=False)
        except:
            print('There was an error running the powerflow! pp.runpp() didnt work')

    def calculate_reward(self, eps=0.01):
        """Calculate the reward associated with a power flow result.

        We count zero flow through the line as when the power flowing into the
        line is equal to the power lost in it. This gives a positive reward.

        A cost (negative reward) is incurred for running the batteries, based
        on the capital cost of the battery and the expected lifetime (currently
        hardcoded to 1000 cycles). So, if the capital cost of the battery is set
        to zero, then producing or consuming power with the battery is free to
        use.

        Parameters
        ----------
        eps: float
            Tolerance

        Attributes
        ----------
        reward_val: The value of the reward function is returned.
        """

        self.reward_val = 0.0
        for i in range(self.net.line.shape[0]):
            cond1a = np.abs(self.net.res_line.p_to_kw.values[i] -
                            self.net.res_line.pl_kw.values[i]) < eps
            cond1b = np.abs(self.net.res_line.p_from_kw.values[i] -
                            self.net.res_line.pl_kw.values[i]) < eps
            check1 = (cond1a or cond1b)
            cond2a = np.abs(self.net.res_line.q_to_kvar.values[i] -
                            self.net.res_line.ql_kvar.values[i]) < eps
            cond2b = np.abs(self.net.res_line.q_from_kvar.values[i] -
                            self.net.res_line.ql_kvar.values[i]) < eps
            check2 = (cond2a or cond2b)
            if check1 and check2:
                self.reward_val += self.net_zero_reward

        # Costs for running batteries
        cap_costs = self.net.storage.cap_cost
        max_e = self.net.storage.max_e_kwh
        min_e = self.net.storage.min_e_kwh
        betas = cap_costs / (2 * 1000 * (max_e - min_e))
        incurred_costs = betas * np.abs(self.net.storage.p_kw)
        for c in incurred_costs:
            self.reward_val -= c
        return self.reward_val
