import pandapower as pp


def get_net(config):
    if config.env_name == 'Six_Bus_POC':
        return six_bus(config.vn_high, config.vn_low, config.length_km,
                       config.std_type)


def six_bus(vn_high=20, vn_low=0.4, length_km=0.03, std_type='NAYY 4x50 SE'):
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
    # create grid connection
    pp.create_ext_grid(net, 0)
    # create lines
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
    return net
