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
            net = self.pp_net(network_name)
        else: 
            self.pp_net = getattr(pp.networks,network_name)
            net = self.pp_net()
            

        self.num_bus = net.bus.shape[0] # How many buses are there
        self.loadbuses = net.load.bus.values # Which ones have loads on them
        self.num_loadbus = net.load.bus.shape[0] # How many load buses are there

        self.load_realpowers = np.zeros(self.num_loadbus)
        self.load_reactivepowers = np.zeros(self.num_loadbus)
        
        self.genbuses = []
        self.gen_real = []
        self.num_sgen = net.sgen.shape[0] # Number of static generators (our simple model of a generator)
        if self.num_sgen > 0:
            for i in range(self.num_sgen):
                self.genbuses = np.append(self.genbuses,net.sgen.bus[i])
                self.gen_real = np.append(self.gen_real,0.0) # Had problems with data type when there were multiple gens vs. one
        

#         self.gen_reactive = 0 # No reactive power yet
        self.num_normalgen = net.gen.shape[0]

        self.reward_val = 0
        
        self.num_lines = net.line.shape[0]
        self.p_lineflow = 0
        self.q_lineflow = 0
        
        # Batteries
        self.num_batteries = 0
        self.soc = []
        self.battery_buses = []
        self.p_batteries = []
        self.energy_capacities = []
        
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
    
    def add_battery(self,bus_number,init_p,init_energy_capacity,init_soc):
        
        self.num_batteries += 1
        self.battery_buses = np.append(self.battery_buses,bus_number)
        self.p_batteries = np.append(self.p_batteries,init_p)
        self.energy_capacities = np.append(self.energy_capacities,init_energy_capacity)
        self.soc = np.append(self.soc,init_soc)
        
        return
    
    def update_batteries(self,battery_powers,dt):
        
        # This is the action
        
        self.p_batteries = battery_powers
        self.soc = np.clip(self.soc + battery_powers*dt, 0, self.energy_capacities)
        
        return
        
        
    def run_powerflow(self,check=False,zero_test = False):
    
        net = self.pp_net()
        
        # Apply loads
        p_load = self.load_realpowers
        q_load = self.load_reactivepowers
        
        df = net.load
        for i in range(self.num_loadbus):
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'p_kw'] = p_load[i]
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'q_kvar'] = q_load[i]

        net.load = df
        
        # Zero out the initialized q generation values from the standard generators so that we only have ones we control (not sure if this works or is how to do it, but I will work on it after Monday)
        if zero_test == True:
            num_real_gen = net.gen.shape[0]
            net.gen.min_q_var = 0 
            net.gen.max_q_var = 0
            for i in range(self.num_normalgen):
                net.gen.p_kw[i] = 0.0
                net.gen.sn_kva[i] = 0.1

        # Apply generation values
        for i in range(self.num_sgen):
            pandapower.create_sgen(net, self.genbuses[i], self.gen_real[i]) # No reactive power yet
        # Apply storage information
        for i in range(self.num_batteries):
            pandapower.create_storage(net,self.battery_buses[i],self.p_batteries[i],self.energy_capacities[i],soc_percent = self.soc[i]/self.energy_capacities[i])
        
        # Run powerflow
        try:
            pp.runpp(net,enforce_q_lims = True)
        except:
            print('There was an error running the powerflow! pp.runpp() didnt work')
        
        # Collect results
        self.p_lineflow = net.res_line.p_to_kw
        self.q_lineflow = net.res_line.q_to_kvar
        # Could also measure the losses
        
        if check == True:  # If check == True then inspect all these objects that are local to this function
            print('Loads everywhere: ',net.res_load)
            print('Res_bus: ',net.res_bus)
            print('Res_line: ',net.res_line)
            print('Net res_gen:', net.res_gen)
            print('Net res sgen: ',net.res_sgen)
        
            print('Lines in: ',net.res_line.p_to_kw)
            print('Lines out: ',net.res_line.p_from_kw)
        return
    
    def calc_reward(self, eps = 0.01):
        
        for i in range(num_lines):
            if np.abs(self.p_lineflow[i]) < eps : 
                self.reward_val += 1.0
                
        return 
        
        
        
        
        
        