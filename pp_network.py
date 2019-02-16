import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks


class net_model:
    
    # The state of the network, powerflow, and reward
    
    def __init__(self,network_name):
        
        self.network_name = network_name
        if network_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
            self.pp_net = pp.networks.create_synthetic_voltage_control_lv_network(network_class=str(network_name))
        else: 
            self.pp_net = getattr(pp.networks,network_name)
            
        net = self.pp_net()
        self.num_bus = net.bus.shape[0] # How many buses are there
        self.loadbuses = net.load.bus.values # Which ones have loads on them
        self.num_loadbus = net.load.bus.shape[0] # How many load buses are there

        self.load_realpowers = np.zeros(self.num_loadbus)
        self.load_reactivepowers = np.zeros(self.num_loadbus)
        
        self.genbuses = []
        self.num_sgen = net.sgen.shape[0] # Number of static generators (our simple model of a generator)
        self.gen_real = 0
#         self.gen_reactive = 0 # No reactive power yet
        self.num_normalgen = net.gen.shape[0]
        
        self.reward_val = 0
        
        self.num_lines = net.line.shape[0]
        self.p_lineflow = 0
        self.q_lineflow = 0
        
        
        return
        
        
    def add_generation(self,bus_number, init_real_power):
        
        # Add where this generation unit is to the list
        self.genbuses = np.append(self.genbuses,bus_number)
        self.num_sgen = self.genbuses.shape[0]
        # Add this generation units value to the list
        generation = np.append(self.gen_real,init_real_power)
        self.gen_real = generation
        
        return

        
    def update_loads(self,new_p, new_q):
        
        self.load_realpowers = new_p
        self.load_reactivepowers = new_q
        
        return

    def update_generation(self,new_gen_p, new_gen_q = 0):
        
        self.genreal = new_gen_p
        
        return
        
    def run_powerflow(self):
    
        net = self.pp_net()
        
        # Apply loads
        p_load = self.load_realpowers
        q_load = self.load_reactivepowers
        
        df = net.load
        for i in range(self.num_loadbus):
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'p_kw'] = p_load[i]
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'q_kvar'] = q_load[i]

        net.load = df
        
        # Apply generation values
        for i in range(self.num_sgen):
            pandapower.create_sgen(net, self.genbuses[i], self.gen_real[i]) # No reactive power yet
        
        # Run powerflow
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except:
            print('There was an error running the powerflow! pp.runpp() didnt work')
        
        # Collect results
        self.p_lineflow = net.res_line.p_to_kw
        self.q_lineflow = net.res_line.q_to_kvar
        # Could also measure the losses
        
        return
    
    def calc_reward(self, eps = 0.01):
        
        for i in range(num_lines):
            if np.abs(self.p_lineflow[i]) < eps : 
                self.reward_val += 1.0
                
        return 
        
        
        
        
        
        