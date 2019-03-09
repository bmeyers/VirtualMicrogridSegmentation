import pandapower as pp
import numpy as np


def get_net(config):
    if config.env_name == 'Six_Bus_POC':
        return six_bus(config.vn_high, config.vn_low, config.length_km,
                       config.std_type, config.battery_locations, config.init_soc,
                       config.energy_capacity, config.static_feeds)


def add_battery(net, bus_number, p_init, energy_capacity, init_soc=0.5,
                max_p=50, min_p=-50, eff=1.0, capital_cost=0, min_e=0.):
    """Change the network by adding a battery / storage unit. """
    pp.create_storage(net, bus_number, p_init, energy_capacity,
                      soc_percent=init_soc, max_p_kw=max_p, min_p_kw=min_p,
                      min_e_kwh=min_e)
    # if 'eff' not in net.storage.columns:
    #     net.storage['eff'] = eff
    # else:
    idx = net.storage.index[-1]
    net.storage.loc[idx, 'eff'] = eff
    # if 'cap_cost' not in net.storage.columns:
    #     net.storage['cap_cost'] = capital_cost
    # else:
    idx = net.storage.index[-1]
    net.storage.loc[idx, 'cap_cost'] = capital_cost


def six_bus(vn_high=20, vn_low=0.4, length_km=0.03, std_type='NAYY 4x50 SE', battery_locations=[3, 6], init_soc=0.5,
            energy_capacity=20.0, static_feeds=None):
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
                    init_soc=init_soc_here)

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


