# Fleet-coordination
This repository contains the work for the project Fleet Robot Coordination using Non-Linear Model Predictive Control.

## Install dependencies
To solve the optimization problem in the MPC-formulation the open source solver [OpEn](https://alphaville.github.io/optimization-engine/docs/installation) is used. The solver is implemented using Rust which can be installed by following the guide on their webpage [Rust](https://www.rust-lang.org/tools/install). 

Anaconda will be used as the package manager of this project, install anaconda according to the instructions on their webpage [Anaconda](https://www.anaconda.com/products/individual). Clone this repository and create the conda enviroment according to the following commands. 

```
conda env create -f env/env.yml
```
