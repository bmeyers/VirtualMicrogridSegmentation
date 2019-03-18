from virtual_microgrids.configs.six_bus_poc import ConfigSixBusPOC
from virtual_microgrids.configs.standard_lv_network import StandardLVNetwork


def get_config(env_name, baseline=True, algorithm='ddpg'):
    """Given an environment name and the baseline option, return the configuration."""
    if env_name == 'Six_Bus_POC':
        return ConfigSixBusPOC(baseline, algorithm)
    if env_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
        return StandardLVNetwork(env_name, baseline, algorithm)

