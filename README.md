# Fleet-coordination
This project aimed to control a fleet of mobile wheeled robots using Non-Linear Model Predictive Control. The individual trajectories for each robots where assumed to be given and they should be altered such that there are no collisions with other robots, static- and dynamic obstacles. Both a centralized and a distributed scheme where compared and both showed promising results. 


![img](docs/displayimg.png)

## Install dependencies
To solve the optimization problem in the MPC-formulation the open source solver [OpEn](https://alphaville.github.io/optimization-engine/docs/installation) is used. The solver is implemented using Rust which can be installed by following the guide on their webpage [Rust](https://www.rust-lang.org/tools/install). 

Anaconda will be used as the package manager of this project, install anaconda according to the instructions on their webpage [Anaconda](https://www.anaconda.com/products/individual). Clone this repository and create the conda enviroment according to the following commands. 

```
conda env create -f env/env_slim.yml
```

## Run the system 


The first step run the system is to activate the enviroment. 

```
conda activate fleet
```

The second step is to build the solvers, assuimg you are in the root of the system run the following commands.
```
cd code 
python mpcgenerator.py
```

Lastly, run the the actual simulation by running the following command.
```
python simulator.py
```

Follow the instructions in the terminal, first choose between my centralized and distributed and then choose between the 5 cases.
