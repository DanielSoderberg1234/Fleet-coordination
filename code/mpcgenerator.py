import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model
from itertools import combinations

"""
    A file that generates the MPC formulation.
"""
class MPCGenerator: 
    def __init__(self, nr_of_robots): 
        self.name = "Fleet-collison"
        self.nr_of_robots = nr_of_robots

    def cost_state_ref(self,x,y,theta,xref,yref,thetaref,q,qtheta): 
        # Cost for deviating from the current reference
        return q*( (xref-x)**2 + (yref-y)**2 ) + qtheta*(thetaref-theta)**2

    def cost_deviation_ref(self,robots,i,j,q,qtheta):
        nu = 2
        nx = 3
        cost = 0

        # Loop over all robots 
        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            ref = robots[robot_id]['Ref']

            # Get the data for the current steps
            refi = ref[i:i+nx]

            # Get the references explicit
            xref, yref, thetaref = refi[0], refi[1], refi[2]
            cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,q,qtheta)
        
        return cost

    def update_robot_states(self,robots,i,j,ts): 
        nu = 2
        nx = 3

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            uj = u[j:j+nu]
            x,y,theta = model(x,y,theta,uj,ts)
            robots[robot_id]['State'] = [x,y,theta]


    def cost_robot2robot_dist(self,x1,y1,x2,y2,qobs): 
        # Cost for being closer than r to the other robot
        return qobs*cs.fmax(0.0, 1.0**2 - (x1-x2)**2 - (y1-y2)**2)

    def cost_collision(self,robots, qobs):
        cost = 0 
        # Iterate over all pairs of robots
        for comb in combinations(range(0,self.nr_of_robots),2):
            x1,y1,theta1 = robots[comb[0]]['State']
            x2,y2,theta2 = robots[comb[1]]['State']
            cost += self.cost_robot2robot_dist(x1,y1,x2,y2,qobs)

        return cost
        
    def cost_control_action(self,u,r): 
        # Cost for the control action
        return r*cs.dot(u, u)

    def cost_all_control_action(self,robots,i,j,r): 
        nu = 2
        nx = 3
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            uj = u[j:j+nu]
            cost += self.cost_control_action(uj,r)
        
        return cost


    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        # But hard constraints on the velocities of the robot
        N = self.nr_of_robots*N
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 3, 20, 0.1)

        # Input vector 2 trajectories, N long with nx states in each i=0,1,2,..,N-1 and the 6 last are the weights
        p = cs.SX.sym('p',self.nr_of_robots*nx*(N+1)+6)

        # Optimization variables 2 robots each with nu control inputs for N steps
        u = cs.SX.sym('u',self.nr_of_robots*nu*N)

        # Dictionary to hold all robot data
        robots = {}

        # Fill the dictionary
        for i in range(0,self.nr_of_robots): 
            # Values to fill the dictionary
            ref_r = p[(N+1)*nx*i:(N+1)*nx*(i+1)]
            u_r = u[nu*N*i:nu*N*(i+1)]
            x_r,y_r,theta_r = ref_r[0], ref_r[1], ref_r[2]

            # All data for robot i, current state, control inputs for all states, reference for all states
            robots[i] = {"State": [x_r,y_r,theta_r], 'u': u_r, 'Ref': ref_r}

        
        # Get weights from input vectir as the last elements
        q, qtheta, r, qN, qthetaN,qobs = p[-6],p[-5],p[-4],p[-3],p[-2],p[-1]

        # Define the cost
        cost = 0
       
        for i,j in zip( range(0,nx*N,nx), range(0,nu*N,nu)): 
            # Calculate the cost of all robots deviating from their reference
            cost += self.cost_deviation_ref(robots,i,j,q,qtheta)
            # Calculate the cost on all control actions
            cost += self.cost_all_control_action(robots,i,j,r)
            # Update the states
            self.update_robot_states(robots,i,j,ts)
            # Calculate the cost of colliding
            cost += self.cost_collision(robots, qobs)  
            

        # Cost for deviating from final reference points
        cost += self.cost_deviation_ref(robots,nx*N,nu*N,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=0.0,vmax=1.5,wmin=-1,wmax=1,N=N)

        return u,p,cost,bounds

    def build_mpc(self): 
        u,p,cost,bounds = self.generate_mpc_formulation()
       
        problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

        build_config = og.config.BuildConfiguration()\
            .with_build_directory("collision_avoidance")\
            .with_build_mode("debug")\
            .with_tcp_interface_config()

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("robot_{}_solver".format(self.nr_of_robots))

        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-5)\
        
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                meta,
                                                build_config,
                                                solver_config)
        builder.build()
       

if __name__=='__main__':
    mpc = MPCGenerator(nr_of_robots=2)
    mpc.build_mpc()