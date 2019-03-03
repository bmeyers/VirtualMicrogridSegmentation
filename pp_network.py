import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks


class net_model:
    
    # The state of the network, powerflow, and reward
    
    def __init__(self,network_name,zero_out_gens = False):
        
        self.network_name = network_name
        if network_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
            self.pp_net = pp.networks.create_synthetic_voltage_control_lv_network(network_class = network_name)
            net = self.pp_net
        else: 
            self.pp_net = getattr(pp.networks,network_name)
            net = self.pp_net()
            

        self.num_bus = net.bus.shape[0] # How many buses are there
        self.loadbuses = net.load.bus.values # Which ones have loads on them
        self.num_loadbus = net.load.bus.shape[0] # How many load buses are there

        self.load_realpowers = np.zeros(self.num_loadbus)
        self.load_reactivepowers = np.zeros(self.num_loadbus)
        
        # Number of static generators (our simple model of a generator)
        self.sgen_buses = []
        self.sgen_real = []
        self.sgen_react = []
        self.num_sgen = net.sgen.shape[0] 
        self.num_sgen_original = net.sgen.shape[0]
        if self.num_sgen > 0:
            for i in range(self.num_sgen):
                self.sgen_buses = np.append(self.sgen_buses,net.sgen.bus[i])
                self.sgen_real = np.append(self.sgen_real,0.0) 
                self.sgen_react = np.append(self.sgen_react,0.0) 
        
        self.zero_out_gens = zero_out_gens
        self.gen_buses = []
        self.gen_real = []
        self.gen_react = []
        self.num_gen = net.gen.shape[0]
        if self.num_gen > 0:
            for i in range(self.num_gen):
                self.gen_buses = np.append(self.gen_buses,net.gen.bus[i])
                if zero_out_gens:
                    self.gen_real = np.append(self.gen_real,0.0) 
                    self.gen_react = np.append(self.gen_react,0.0) 
                        
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
        
    def add_sgeneration(self,bus_number,init_real_power,init_react_power = 0.0):
        
        # Add where this generation unit is to the list
        self.sgen_buses = np.append(self.sgen_buses, bus_number)
        self.num_sgen = self.sgen_buses.shape[0]
        # Add this generation units value to the list
        self.sgen_real = np.append(self.sgen_real, init_real_power)
        self.sgen_react = np.append(self.sgen_react, init_react_power)
        
        return
    
    def add_generation():
        
        print('Add functionality to have a non static generator added')
        
        return

        
    def update_loads(self,new_p, new_q):
        
        self.load_realpowers = new_p
        self.load_reactivepowers = new_q
        
        return
    

    def update_generation(self,new_gen_p, new_gen_q = False):
        
        self.sgen_real = new_gen_p
        if new_gen_q:
            self.sgen_react = new_gen_q
        
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
        
        
    def run_powerflow(self,check=False):
    
        if self.network_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
            net = self.pp_net
        else:
            net = self.pp_net()
        
        # Apply loads
        p_load = self.load_realpowers
        q_load = self.load_reactivepowers
        
        df = net.load
        for i in range(self.num_loadbus):
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'p_kw'] = p_load[i]
            df.loc[lambda df: df['bus'] == self.loadbuses[i], 'q_kvar'] = q_load[i]

        net.load = df
        
 
        if self.num_gen > 0:
            net.gen.min_q_var = 0 
            net.gen.max_q_var = np.max(self.gen_react)
            net.gen.p_kw = self.gen_real # Should be negative
            net.gen.sn_kva = np.sqrt(np.power(self.gen_real, 2) + np.power(self.gen_react, 2)) # Want to set p and q, not s

        # Apply generation values to static generators --- add functionality for other generators still
        for i in range(self.num_sgen):
            pandapower.create_sgen(net, self.sgen_buses[i], self.sgen_real[i], self.sgen_react[i])
        # Apply storage information
        for i in range(self.num_batteries):
            pandapower.create_storage(net, self.battery_buses[i], self.p_batteries[i], 
                                      self.energy_capacities[i], soc_percent = self.soc[i] / self.energy_capacities[i])
        
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
        
        
        
        
        
        