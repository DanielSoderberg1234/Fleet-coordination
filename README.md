# Fleet-coordination
This project aimed to control a fleet of mobile wheeled robots using Non-Linear Model Predictive Control. The individual trajectories for each robots where assumed to be given and they should be altered such that there are no collisions with other robots, static- and dynamic obstacles. Both a centralized and a distributed schemes where compared and both showed promising results. The solve times for each iteration and a number of different cases can be seen in the table. The complexity of the centralized approach wrt to the number of robots points to being linear while for the distributed case the complexity seems to be constant.

![solve](docs/solvetimes.PNG)

The figure below shows an example of the collision avoidance scheme using the centralized approach, there are 10 robots represented as rectangles and a dynamic obstacle represented as an epplipse in this case. Videos of more cases can be found on [YouTube](https://www.youtube.com/playlist?list=PLjko-_vToC0wUDgxAlXtwgOw34NESLCbh)

![img](docs/displayimg.png)

## Install dependencies
To solve the optimization problem in the MPC-formulation the open source solver [OpEn](https://alphaville.github.io/optimization-engine/docs/installation) is used. To be able to use the solver a few installations are needed. For Windows, begin by installing [Visual Studio](https://visualstudio.microsoft.com/downloads/) with the Microsoft C++ build tools, this is needed for Rust- and C-code that is automatically generated. Install Rust by following the guide on their webpage [Rust](https://www.rust-lang.org/tools/install). 

To install the necessary dependencies including [OpEn](https://alphaville.github.io/optimization-engine/docs/installation) we have used Anaconda as the package manager of this project. Install anaconda according to the instructions on their webpage [Anaconda](https://www.anaconda.com/products/individual). To run the code, clone this repository and create the conda enviroment according to the following commands, assuming you are in the root of this repository. 

```
conda env create -f env/env.yml
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
