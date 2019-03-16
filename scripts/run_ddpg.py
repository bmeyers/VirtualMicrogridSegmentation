import sys
sys.path.append('..')

from virtual_microgrids.configs import get_config
from virtual_microgrids.powerflow import NetModel
from virtual_microgrids.algorithms import DDPG
from virtual_microgrids.utils.general import get_logger

if __name__ == '__main__':
    config = get_config('Six_Bus_POC', 'DDPG')
    env = NetModel(config=config)
    # train model
    model = DDPG(env, config)
    model.run()