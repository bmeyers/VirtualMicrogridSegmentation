import numpy as np
import pandas as pd
import pandapower as pp
from copy import deepcopy
from virtual_microgrids.configs import get_config
from virtual_microgrids.powerflow.network_generation import get_net
from virtual_microgrids.utils import Graph


class NetModel(object):
    """Building and interacting with a network model to simulate power flow.

    In this class we model all of the network component including loads,
    generators, batteries, lines, buses, and transformers. The state of each is
    tracked in a pandapower network object.
    """
    def __init__(self, config=None, env_name='Six_Bus_POC', baseline=True):
        """Initialize attributes of the object and zero out certain components
        in the standard test network."""

        if config is not None:
            self.config = config
            self.net = get_net(self.config)
        else:
            self.config = get_config(env_name, baseline)
            self.net = get_net(self.config)

        self.reward_val = 0.0

        self.tstep = self.config.tstep
        self.net_zero_reward = self.config.net_zero_reward
        self.initial_net = pp.copy.deepcopy(self.net)
        self.time = 0
        self.n_load = len(self.net.load)
        self.n_sgen = len(self.net.sgen)
        self.n_gen = len(self.net.gen)
        self.n_storage = len(self.net.storage)
        self.observation_dim = self.n_load + self.n_sgen + self.n_gen + 2 * self.n_storage
        self.action_dim = self.n_gen + self.n_storage
        self.graph = Graph(len(self.net.bus))
        for idx, entry in self.net.line.iterrows():
            self.graph.addEdge(entry.from_bus, entry.to_bus)

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
        reward = self.calculate_reward(eps=self.config.reward_epsilon)
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

    def calculate_reward(self, eps=0.001, type=4):
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
        c1 = np.abs(self.net.res_line.p_to_kw - self.net.res_line.pl_kw) < eps
        c2 = np.abs(self.net.res_line.p_from_kw - self.net.res_line.pl_kw) < eps
        zeroed_lines = np.logical_or(c1.values, c2.values)
        # Type 1 Reward: count of lines with zero-net-flow
        if type == 1:
            self.reward_val = np.sum(zeroed_lines, dtype=np.float)
        # Type 2 Reward: count of nodes not pulling power from grid
        elif type in [2, 3, 4]:
            graph_new = deepcopy(self.graph)
            for line_idx, zeroed in enumerate(zeroed_lines):
                if zeroed:
                    v = self.net.line.from_bus[line_idx]
                    w = self.net.line.to_bus[line_idx]
                    graph_new.removeEdge(v, w)
            self.reward_val = 0
            ext_connections = self.net.ext_grid.bus.values
            num_vmgs = 0
            for subgraph in graph_new.connectedComponents():
                if not np.any([item in subgraph for item in ext_connections]):
                    self.reward_val += len(subgraph)
                    num_vmgs += 1
            self.reward_val *= num_vmgs
        elif type == 5:
            pass

        # Add distance function:
        if type == 3:
            line_flow_values = np.maximum(np.abs(self.net.res_line.p_to_kw),
                                          np.abs(self.net.res_line.p_from_kw)) - self.net.res_line.pl_kw
            self.reward_val -= self.config.cont_reward_lambda * np.linalg.norm(line_flow_values, 1)
        elif type == 4:
            line_flow_values = np.maximum(np.abs(self.net.res_line.p_to_kw),
                                          np.abs(self.net.res_line.p_from_kw)) - self.net.res_line.pl_kw
            self.reward_val -= self.config.cont_reward_lambda * np.sum(np.minimum(np.abs(line_flow_values),
                                                                                  1.0*np.ones(np.shape(line_flow_values)[0])))
        # Costs for running batteries
        cap_costs = self.net.storage.cap_cost
        max_e = self.net.storage.max_e_kwh
        min_e = self.net.storage.min_e_kwh
        betas = cap_costs / (2 * 1000 * (max_e - min_e))
        incurred_costs = betas * np.abs(self.net.storage.p_kw)
        for c in incurred_costs:
            self.reward_val -= c
        return self.reward_val

if __name__ == "__main__":
    env1 = NetModel(env_name='Six_Bus_POC')
    env1.step([-0.02, -0.02])
