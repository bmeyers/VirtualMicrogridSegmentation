import sys
sys.path.append('..')

from virtual_microgrids.configs import get_config
from virtual_microgrids.powerflow import NetModel
from virtual_microgrids.algorithms import PG

if __name__ == '__main__':
    config = get_config('Six_Bus_POC', algorithm='PG')
    env = NetModel(config=config)
    # train model
    model = PG(env, config)
    model.run()