# VirtualMicrogridSegmentation
CS234 Project, Winter 2019


## Overall structure of the code / implementation: 

First everything is initialized

On the network side: 
- Solve the powerflow
- Calculate the new reward function value

On the agent side: 
- Observe the reward function value
- Policy improvement
- Select and apply new action


These two pieces work together to form the learning loop. 


## Code components
