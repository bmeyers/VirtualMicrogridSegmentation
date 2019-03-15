import pandapower as pp
import numpy as np
from pandapower.networks import create_synthetic_voltage_control_lv_network as mknet


def get_net(config):
    """Given the configuration, call a function to create the network object."""
    if config.env_name == 'Six_Bus_POC':
        return six_bus(config.vn_high, config.vn_low, config.length_km,
                       config.std_type, config.battery_locations, config.init_soc,
                       config.energy_capacity, config.static_feeds, config.gen_locations,
                       config.gen_p_max, config.gen_p_min, config.storage_p_max,
                       config.storage_p_min)
    if config.env_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
        return standard_lv(config.env_name, config.remove_q, config.static_feeds_new, config.clear_loads_sgen,
                           config.clear_gen, config.battery_locations, config.percent_battery_buses,
                           config.batteries_on_leaf_nodes_only, config.init_soc, config.energy_capacity,
                           config.gen_locations, config.gen_p_max, config.gen_p_min, config.storage_p_max,
                           config.storage_p_min)


def add_battery(net, bus_number, p_init, energy_capacity, init_soc=0.5,
                max_p=50, min_p=-50, eff=1.0, capital_cost=0, min_e=0.):
    """Change the network by adding a battery / storage unit.

    This function creates a storage element in net, and adds two non-standard columns: efficiency and capital cost.

    Parameters
    ----------
    net: class
        The pandapower network model
    bus_number: int
        Where the battery will be added
    p_init: float
        The power draw / input of the battery on initialization
    init_soc: float
        The state of charge
    max_p: float
        The max rate that power can be drawn by the battery
    min_p: float
        The max rate that power can be pulled from the battery (negative).
    eff: float
        The efficiency
    capital_cost: float
        The capital cost of the battery
    min_e: float
        The minimum energy in the battery
    """
    pp.create_storage(net, bus_number, p_init, energy_capacity,
                      soc_percent=init_soc, max_p_kw=max_p, min_p_kw=min_p,
                      min_e_kwh=min_e)
    idx = net.storage.index[-1]
    net.storage.loc[idx, 'eff'] = eff
    net.storage.loc[idx, 'cap_cost'] = capital_cost


def six_bus(vn_high=20, vn_low=0.4, length_km=0.03, std_type='NAYY 4x50 SE', battery_locations=[3, 6], init_soc=0.5,
            energy_capacity=20.0, static_feeds=None, gen_locations=None, gen_p_max=0.0, gen_p_min=-50.0,
            storage_p_max=50.0, storage_p_min=-50.0):
    """This function creates the network model for the 6 bus POC network from scratch.

    Buses and lines are added to an empty network based on a hard-coded topology and parameters from the config file
    (seen as inputs). The only controllable storage added in this network are batteries, and the input static_feeds is
    used to add loads and static generators which are not controlled by the agent. The first value in the series is
    taken for initialization of those elements.
    """
    net = pp.create_empty_network(name='6bus', f_hz=60., sn_kva=100.)
    # create buses
    for i in range(8):
        nm = 'bus{}'.format(i)
        if i == 0:
            pp.create_bus(net, name=nm, vn_kv=vn_high)
        elif i == 1:
            pp.create_bus(net, name=nm, vn_kv=vn_low)
        else:
            if i <= 4:
                zn = 'Side1'
            else:
                zn = 'Side2'
            pp.create_bus(net, name=nm, zone=zn, vn_kv=vn_low)
    #  create grid connection
    pp.create_ext_grid(net, 0)
    #  create lines
    pp.create_line(net, 0, 1, length_km=length_km, std_type=std_type,
                   name='line0')
    pp.create_line(net, 1, 2, length_km=length_km, std_type=std_type,
                   name='line1')
    pp.create_line(net, 2, 3, length_km=length_km, std_type=std_type,
                   name='line2')
    pp.create_line(net, 2, 4, length_km=length_km, std_type=std_type,
                   name='line3')
    pp.create_line(net, 1, 5, length_km=length_km, std_type=std_type,
                   name='line4')
    pp.create_line(net, 5, 6, length_km=length_km, std_type=std_type,
                   name='line5')
    pp.create_line(net, 5, 7, length_km=length_km, std_type=std_type,
                   name='line6')

    #  add controllable storage
    for idx, bus_number in enumerate(battery_locations):
        energy_capacity_here = energy_capacity
        init_soc_here = init_soc
        if np.size(energy_capacity) > 1:
            energy_capacity_here = energy_capacity[idx]
        if np.size(init_soc) > 1:
            init_soc_here = init_soc[idx]

        add_battery(net, bus_number=bus_number, p_init=0.0, energy_capacity=energy_capacity_here,
                    init_soc=init_soc_here, max_p=storage_p_max, min_p=storage_p_min)

    # Add controllable generator
    if gen_locations is not None:
        for idx, bus_number in enumerate(gen_locations):
            pp.create_gen(net, bus_number, p_kw=0.0, min_q_kvar=0.0, max_q_kvar=0.0, min_p_kw=gen_p_min,
            max_p_kw=gen_p_max)

            ##### TODO : Have different limits for different generators and storage #####

    #  add loads and static generation
    if static_feeds is None:
        print('No loads or generation assigned to network')
    else:
        if len(static_feeds) > 0:
            for key, val in static_feeds.items():
                init_flow = val[0]
                print('init_flow: ', init_flow, 'at bus: ', key)
                if init_flow > 0:
                    pp.create_load(net, bus=key, p_kw=init_flow, q_kvar=0)
                else:
                    pp.create_sgen(net, bus=key, p_kw=init_flow, q_kvar=0)

    return net


def standard_lv(env_name, remove_q=True, static_feeds_new=None, clear_loads_sgen=False, clear_gen=True,
                battery_locations=None, percent_battery_buses=0.5, batteries_on_leaf_nodes_only=True, init_soc=0.5,
                energy_capacity=20.0, gen_locations=None, gen_p_max=0.0, gen_p_min=-50.0,
            storage_p_max=50.0, storage_p_min=-50.0):
    """This function creates a network model using the set of synthetic voltage control low voltage (LV) networks from
    pandapower.

    The environment name, env_name, chooses which of the models to create out of 'rural_1', 'rural_2', 'village_1',
    'village_2', and 'suburb_1'.

    Then options can be triggered to remove all reactive power components from the network (as we do in this project),
    or to remove static generators, loads, and generators that come with the standard model of the network. New
    batteries and generators are added which will be used as controllable resources by the agent.

    Static_feeds is a dictionary used by other functions to define the state of the network as we step through time, and
    contains the power values of the non-controllable elements: static generators and loads. In this method we use
    static_feeds_new, a subset of static_feeds, to create new loads and static generators in the network that did not
    ship with the model.
    """

    net = mknet(network_class=env_name)

    # Remove q components
    if remove_q:
        net.load.q_kvar = 0
        net.sgen.q_kvar = 0
        net.gen.q_kvar = 0
        net.gen.min_q_kvar = 0
        net.gen.max_q_kvar = 0
        net.shunt.in_service = False

    # Remove built in loads and generators
    if clear_loads_sgen:
        net.load.in_service = False
        net.sgen.in_service = False
    if clear_gen:
        net.gen.in_service = False
    net.storage.in_service = False

    #  add controllable storage
    if battery_locations is not None:
        applied_battery_locations = battery_locations
    elif percent_battery_buses > 0:
        if batteries_on_leaf_nodes_only:
            leaf_nodes = []
            for i in net.line.to_bus.values:
                if i not in net.line.from_bus.values:
                    leaf_nodes.append(i)
            applied_battery_locations = np.random.choice(leaf_nodes, int(percent_battery_buses * len(leaf_nodes)),
                                                         replace=False)
        else:
            applied_battery_locations = np.random.choice(net.bus.shape[0],
                                                         int(percent_battery_buses * net.bus.shape[0]), replace=False)
    if len(applied_battery_locations) > 0:
        num_batteries = len(applied_battery_locations)
        for idx, bus_number in enumerate(applied_battery_locations):
            energy_capacity_here = energy_capacity
            init_soc_here = init_soc
            if np.size(energy_capacity) > 1:
                energy_capacity_here = energy_capacity[0]
                if np.size(energy_capacity) == num_batteries:
                    energy_capacity_here = energy_capacity[idx]
            if np.size(init_soc) > 1:
                init_soc_here = init_soc[0]
                if np.size(energy_capacity) == num_batteries:
                    init_soc_here = init_soc[idx]
            add_battery(net, bus_number=bus_number, p_init=0.0, energy_capacity=energy_capacity_here,
                        init_soc=init_soc_here, max_p=storage_p_max, min_p=storage_p_min)
    # Add controllable generator
    if gen_locations is not None:
        for idx, bus_number in enumerate(gen_locations):
            pp.create_gen(net, bus_number, p_kw=0.0, min_q_kvar=0.0, max_q_kvar=0.0, max_p_kw=gen_p_max,
                          min_p_kw=gen_p_min)

    if static_feeds_new is None:
        print('No loads or generation added to network')
    else:
        if len(static_feeds_new) > 0:
            for key, val in static_feeds_new.items():
                init_flow = val[0]
                print('init_flow: ', init_flow, 'at bus: ', key)
                if init_flow > 0:
                    pp.create_load(net, bus=key, p_kw=init_flow, q_kvar=0)
                else:
                    pp.create_sgen(net, bus=key, p_kw=init_flow, q_kvar=0)

    #  Name buses for plotting
    for i in range(net.bus.name.shape[0]):
        net.bus.name.at[i] = 'bus' + str(i)

    return net


