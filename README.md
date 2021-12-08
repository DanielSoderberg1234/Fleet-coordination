# Fleet-coordination
This project aimed to control a fleet of mobile wheeled robots using Non-Linear Model Predictive Control. The individual trajectories for each robots where assumed to be given and they should be altered such that there are no collisions with other robots, static- and dynamic obstacles. Both a centralized and a distributed schemes where compared and both showed promising results. 

The formulation of the centralized NMPC can be written as: 


$$ \min_u \sum_{i=0}^{N-1} \bigg[ \sum_{j=1}^M J_{\tau}(\mathbf{x}_{j,i}^{\textit{ref}},\mathbf{x}_{j,i}) + J_{u}(\mathbf{u}_{j,i}^{\textit{ref}},\mathbf{u}_{j,i}) + \\ J_{a}(\mathbf{u}_{j,i+1}, \mathbf{u}_{j,i})+ J_{\mathcal{O}}(\mathbf{p}_{j,i}) + J_{\mathcal{B}}(\mathbf{p}_{j,i}) + \\ J_{dyn}(\mathbf{p}_{j,i},\mathbf{c}_{k,i}) + \sum_{c \in \mathcal{C} } J_{dist}(\mathbf{p}_{c_1,i},\mathbf{p}_{c_2,i})  \bigg] \\ + J_{\tau}(\mathbf{x}_{j,N}^{\textit{ref}},\mathbf{x}_{j,N})  \\ \\ \textrm{s.t.} \quad \mathbf{u}_{min} \leq \mathbf{u}_{i,j} \leq \mathbf{u}_{max} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \\ \mathbf{x}_{j,i+1} = f(\mathbf{x}_{j,i}, \mathbf{u}_{j,i}) \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

![img](docs/displayimg.png)

## Install dependencies
To solve the optimization problem in the MPC-formulation the open source solver [OpEn](https://alphaville.github.io/optimization-engine/docs/installation) is used. The solver is implemented using Rust which can be installed by following the guide on their webpage [Rust](https://www.rust-lang.org/tools/install). 

Anaconda will be used as the package manager of this project, install anaconda according to the instructions on their webpage [Anaconda](https://www.anaconda.com/products/individual). Clone this repository and create the conda enviroment according to the following commands. 

```
conda env create -f env/env_slim.yml
```
