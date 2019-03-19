import sys
sys.path.append('..')

from virtual_microgrids.configs import get_config
from virtual_microgrids.powerflow import NetModel
from virtual_microgrids.algorithms import DDPG

if __name__ == '__main__':
    config = get_config('rural_1', algorithm='DDPG')  # 'Six_Bus_MVP2', algorithm='DDPG')
    env = NetModel(config=config)
    # train model
    model = DDPG(env, config)
    model.run()