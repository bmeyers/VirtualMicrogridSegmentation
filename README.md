# Virtual Microgrid Segmentation
CS234 Project, Winter 2019

Project team: Bennet Meyers and Siobhan Powell

Contact the authors: bennetm or siobhan.powell at stanford dot edu

## Overview
Recent work has shown that microgrids can increase both grid flexibility and grid resiliency to unanticipated outages 
caused by events such as cyber attacks or extreme weather. A subclass of microgrids, known as “virtual 
islands”, occur when sections of a grid operate in isolation without any powerflow between them and the   larger grid, 
despite remaining physically connected. If a grid can can partition into virtual islands in anticipation of an incoming 
resiliency event, customers in those islands will be less likely to experience outages.

The goal of this project is to train a deep reinforcement learning (RL) agent to create and maintain as many small virtual 
islands as possible by operating a grids storage resources. The agent is rewarded for separating nodes from the external
grid connection and for splitting the graphs into as many segments as possible.  

As our environment is deterministic, we implement PG (policy gradient) and DDPG (deep deterministic policy gradient) algorithms to train the agent, and
apply it to a small test network. We find the DDPG performs the best, and it can successfully maintain microgrids even when
the loads are time varying and change between episodes. 

## The DDPG algorithm

The DDPG algorithm was introduced by Lillicrap et al in "Continous control with deep reinforcement learning", available on
arXiv at https://arxiv.org/abs/1509.02971. 

This algorithm builds on the DPG deterministic actor-critic approach proposed by Silver et al in "Deterministic 
Policy Gradient Algorithms", available at http://proceedings.mlr.press/v32/silver14.pdf. DDPG combines this approach with the 
successes of deep learning from DQN. It is model-free, off-policy, and has been shown to learn complex continuous control 
tasks in high dimensions quite well. 

Standard stochastic PG involves taking the expectation over the distribution of actions to calculate the gradient step. 
DDPG simply moves the policy in the direction of the gradient of Q, removing the need for an integral over the action space, 
making it much more efficient at learning in our environment.

In DDPG the algorithm builds a critic network to estimate the state action value function, Q(s,a). An actor network is built to 
learn a behaviour from the critic estimation. The algorithm learns a deterministic policy but implements a stochastic behaviour
policy by adding noise to the action choice to properly explore the solution space. The tuning and scheduling of this exploration 
noise term is crucial to the success of the algorithm. 

To help with convergence and stability, the algorithm is implemented with experience replay and with semi-stationary target
networks. For more information on the theory and the algorithm applied, please refer to the papers.  

## Structure of the Code

There are two main sides to the code: the network and the agents. 

The network is generated using Pandapower (https://pandapower.readthedocs.io/en/v1.6.1/index.html). 

The NetModel class in powerflow/pp_network.py maintains the network 
object throughout the simulation. It controls how the agent can interact with the network 
and with the powerflow simulations with methods to step in time, calculate the reward, reset the network, 
report the state to the agent, and update the network devices. These devices include uncontrollable and controllable devices: 
loads and static generators are set by an uncontrollable unknown feed;  the powers of storage and diesel generators are 
controlled by the agent. 

The initial network is generated by functions in powerflow/network_generation.py using configurations stored
in configs. Each config defines all the parameters behind one test set up, including those of the network and some 
elements of the agent set up.   

The ActorNetwork and CriticNetwork objects are created in agents/actor_network.py and agents/critic_network.py, and the 
DDPG object uses them to learn the optimal policy. DDPG manages the training of the actor/critic networks
and controls the interactions with the grid network model. 


#### Code organization

The main folder contains scratch notebooks for testing, developing, and interacting with the environments.

The 'scripts' folder contains scripts to run the algorithms. For example, change the environment name or config name
in 'run_ddpg.py' and then run

    python run_ddpgy.py 
    
to start the simulation. 

The 'virtual_microgrids' folder contains all the pieces of the simulation. To run you do not need to change anything in here,
but to change parameters or change the algorithm you will need to work with these files.  
- The subfolder 'agents' contains the classes
to build the actor and critic network objects. 
- The 'algorithms' subfolder classes which run the PG and DDPG implementations. 
- The 'configs' subfolder contains the configuration files for each test case and network. To create a new or altered test case,
create a new config file in the style of six_bus_mvp1.py, for example. 
- The 'powerflow' subfolder contains a class to manage the power network and functions to create the networks from the config files
- The 'utils' subfolder contains tools used throughout the other methods and functions, including the schedules used to generate the noise
 

The 'results' folder contains the outputs from running the algorithm. Running the command 

    tensorboard --logdir [path to results folder]
    
and then visiting 

    localhost:6006
    
in your browser will let you inspect the tensorflow setup and see plots of the results.  
