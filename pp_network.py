import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.networks import create_synthetic_voltage_control_lv_network as mknet


class NetModel(object):
    """Building and interacting with a network model to simulate power flow.

    In this class we model all of the network component including loads,
    generators, batteries, lines, buses, and transformers. The state of each is
    tracked in a pandapower network object.
    """
    
    def __init__(self, net_given=None, network_name='rural_1',
                 zero_out_gen_shunt_storage=True):
        """Initialize attributes of the object and zero out certain components
        in the standard test network."""

        if net_given is not None:
            self.net = net_given
            self.network_name = 'custom_network'
        else:
            self.net = mknet(network_class=network_name)
            self.network_name = network_name

        if zero_out_gen_shunt_storage:

            self.net.sgen.p_kw = 0
            self.net.sgen.q_kvar = 0

            self.net.gen.p_kw = 0
            self.net.gen.q_kvar = 0
            self.net.gen.min_p_kw = 0
            self.net.gen.max_p_kw = 0
            self.net.gen.min_q_kvar = 0
            self.net.gen.max_q_kvar = 0
            self.net.gen.sn_kva = 0

            self.net.shunt.p_kw = 0
            self.net.shunt.q_kvar = 0
            self.net.shunt.in_service = False

        self.reward_val = 0

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

    def update_loads(self, new_p, new_q):
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

        self.net.load.p_kw = new_p
        self.net.load.q_kvar = new_q

    def update_static_generation(self, new_sgen_p, new_sgen_q):
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

        self.net.sgen.p_kw = new_sgen_p
        self.net.sgen.q_kvar = new_sgen_q

    def update_generation(self, new_gen_p):
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

        self.net.gen.p_kw = new_gen_p

    def add_battery(self, bus_number, init_p, init_energy_capacity, init_soc):
        """Change the network by adding a battery / storage unit.

        Parameters
        ----------
        bus_number: int
            The bus at which the generator should be added
        init_p: float
            The real power flow to/from the battery for initialization.
        init_energy_capacity: float
            The energy capacity of the battery.
        init_soc: float
            The initial state of charge.

        Attributes
        ----------
        net: object
            The network object is updated
        """

        pp.create_storage(self.net, bus_number, init_p,
                                  init_energy_capacity,
                                  soc_percent=init_soc / init_energy_capacity)
    
    def update_batteries(self, battery_powers, dt):
        """Update the batteries / storage units in the network.

        This method assumes that the orders match, i.e. the order the buses in
        self.net.gen.bus matches where the generation values in new_gen_p
        should be applied based on their indexing.

        Parameters
        ----------
        battery_powers: array_like
            The power flow into / out of each battery, shape (number of traditional generators, 1).
        dt: float
            The time duration of this power flow in hours.

        Attributes
        ----------
        self.net.storage: object
            The storage values in the network object are updated.
        """
        self.net.storage.p_kw = battery_powers
        soc_raw = self.net.storage.soc_percent + (battery_powers * dt / self.net.storage.max_e_kwh)
        self.net.storage.soc_percent = np.clip(soc_raw, 0.0, 1.0)

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
                     calculate_voltage_angles=False, voltage_depend_loads=False)
        except:
            print('There was an error running the powerflow! pp.runpp() didnt work')

    def calculate_reward(self, eps=0.01):
        """Calculate the reward associated with a power flow result.

        We count zero flow through the line as when the power flowing into the
        line is equal to the power lost in it.

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
            cond1a = np.abs(self.net.res_line.p_to_kw.values[i] - self.net.res_line.pl_kw.values[i]) < eps
            cond1b = np.abs(self.net.res_line.p_from_kw.values[i] - self.net.res_line.pl_kw.values[i]) < eps
            check1 = (cond1a or cond1b)
            cond2a = np.abs(self.net.res_line.q_to_kvar.values[i] - self.net.res_line.ql_kvar.values[i]) < eps
            cond2b = np.abs(self.net.res_line.q_from_kvar.values[i] - self.net.res_line.ql_kvar.values[i]) < eps
            check2 = (cond2a or cond2b)
            if check1 and check2:
                self.reward_val += 1.0
