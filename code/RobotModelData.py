from dataclasses import dataclass

@dataclass
class RobotModelData:
    # Robot parameters
    nr_of_robots: int = 2   # Number of robots
    nx: int = 3             # Number of states for each robot
    nu: int = 2             # Nr of control inputs
    N: int = 20             # Length of horizon 

    # Model parameters
    ts: float = 0.1         # Sampling time
    q: float = 10           # Cost of deviating from x and y reference
    qtheta: float = 1       # Cost of deviating from angle reference
    r: float = 0.01         # Cost for control action
    qN: float = 100         # Final cost of deviating from x and y reference
    qthetaN: float = 10     # Final cost of deviating from angle reference
    qobs: float = 200       # Cost for being closer than r to the other robot


    def get_weights(self):
        return [self.q, self.qtheta, self.r, self.qN, self.qthetaN, self.qobs]



