## Deep Reinforcement Learning - DDPG, TD3 and A2C applied to Reacher Environment
<img src='reacher.gif' width="500" height="300">

This repository holds the project code for using Deep Reinforcement Learning algorithms - DDPG, TD3 and A2C on Reacher envrionment with continuous controls provided by Unity Technology. It is part of the Udacity [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) requirement. 

The state is represented as a 33 dimensional observations including the agent's velocity, ray-based perception of objects etc.. The agent has to learn to choose the optimal action based on the state it finds itself in. The action space is a 4 dimensional continuous space representing the torque of the double jointed robot arms. The agent's goal is to track a moving object as many time steps as possible. A reward of +0.1 is earned each step the agent successfully tracks the moving object. The environment is considered solved when the return reaches an average of 30 consistently (over 100 episodes). 

All three algorithms solved the environment and A2C seems to be the best.   
<img src='A2C_reacher.png' width="400" height="250">

## Installation
To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

6. Download the environment - there are two versions of the environment below (specifically built for this project, **not** the Unity ML-Agents package). Then place the file in the root folder and unzip the file.
For the first verion:
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    * Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)    
For the second version:    
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)    
    * Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)    
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)    
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

7. Import the environment in Jupyter notebook under the the *drlnd* environment.
```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="[to be replaced with file below depending on OS]")
```
Replace the file name with the following depending on OS:  
  * Mac: "Reacher.app"
  * Windows (x86): "Reacher_Windows_x86/Reacher.exe"
  * Windows (x86_64): "Reacher_Windows_x86_64/Reacher.exe"
  * Linux (x86): "Reacher_Linux/Reacher.x86"
  * Linux (x86_64): "Reacher_Linux/Reacher.x86_64"
  * Linux (x86, headless): "Reacher_Linux_NoVis/Reacher.x86"
  * Linux (x86_64, headless): "Reacher_Linux_NoVis/Reacher.x86_64"
## How to Run
Load the Jupyter notebook *Report.ipynb* and run all cells.
