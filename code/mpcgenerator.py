import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
#from code.test import cost_all_acceleration
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
        nx = 5
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
        nx = 5

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
        
    def cost_control_action(self,u,u_ref,r): 
        # Cost for the control action
        return r*cs.dot(u_ref-u,u_ref-u)

    def cost_all_control_action(self,robots,i,j,r): 
        nu = 2
        nx = 5
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            refi = robots[robot_id]['Ref'][i:i+nx]
            u_ref = refi[3:]
            uj = u[j:j+nu]
            cost += self.cost_control_action(uj,u_ref,r)
        
        return cost

    def cost_acceleration(self,u0,u1,qaccV,qaccW): 
        cost = 0
        cost += qaccV*(u1[0]-u0[0])**2
        cost += qaccW*(u1[1]-u0[1])**2
        return cost
        

    def cost_all_acceleration(self,robots,qaccV,qaccW): 
        nu = 2
        nx = 5
        N = 20
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']

            u0 = u[:-2]
            u1 = u[2:]

        for i in range(0,N-2,2):
            u0i = u0[i:i+2]
            u1i = u1[i:i+2]
            cost += self.cost_acceleration(u0,u1,qaccV,qaccW)
            
        return cost

    def cost_inital_acceleration(self,robots,qaccV,qaccW): 
        nu = 2
        nx = 5
        N = 20
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            v_ref,w_ref = robots[robot_id]['Init_u']
            u_ref = cs.vertcat(v_ref,w_ref)
            u = robots[robot_id]['u'][:nu]
            cost += self.cost_acceleration(u,u_ref,qaccV,qaccW)
            
        return cost

    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        # But hard constraints on the velocities of the robot
        N = self.nr_of_robots*N
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 5, 20, 0.1)

        # Input vector 2 trajectories, N long with nx states in each i=0,1,2,..,N-1 and the 6 last are the weights
        p = cs.SX.sym('p',self.nr_of_robots*nx*(N+1))

        Q = cs.SX.sym('Q',8)

        # Optimization variables 2 robots each with nu control inputs for N steps
        u = cs.SX.sym('u',self.nr_of_robots*nu*N)

        # Dictionary to hold all robot data
        robots = {}

        # Fill the dictionary
        for i in range(0,self.nr_of_robots): 
            # Values to fill the dictionary
            ref_r = p[(N+1)*nx*i:(N+1)*nx*(i+1)]
            u_r = u[nu*N*i:nu*N*(i+1)]
            x_r,y_r,theta_r,v_ref,w_ref = ref_r[0], ref_r[1], ref_r[2], ref_r[3], ref_r[4]

            # All data for robot i, current state, control inputs for all states, reference for all states
            robots[i] = {"State": [x_r,y_r,theta_r], "Init_u": [v_ref,w_ref], 'u': u_r, 'Ref': ref_r}

        
        # Get weights from input vectir as the last elements
        #q, qtheta, r, qN, qthetaN,qobs = p[-6],p[-5],p[-4],p[-3],p[-2],p[-1]
        q, qtheta, r, qN, qthetaN,qobs, qaccV,qaccW = Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7]

        # Define the cost
        cost = 0
        
        # Penalize the first acceleration
        cost += self.cost_inital_acceleration(robots,qaccV,qaccW)

        for i,j in zip( range(0,nx*N,nx), range(0,nu*N,nu)): 
            # Calculate the cost of all robots deviating from their reference
            cost += self.cost_deviation_ref(robots,i,j,q,qtheta)
            # Calculate the cost on all control actions
            cost += self.cost_all_control_action(robots,i,j,r)
            # Update the states
            self.update_robot_states(robots,i,j,ts)
            # Calculate the cost of colliding
            cost += self.cost_collision(robots, qobs)  
            
        # Add acceleration cost
        cost += self.cost_all_acceleration(robots,qaccV,qaccW)

        # Cost for deviating from final reference points
        cost += self.cost_deviation_ref(robots,nx*N,nu*N,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=0.0,vmax=1.5,wmin=-1,wmax=1,N=N)
        
        # Concate all parameters
        p = cs.vertcat(p,Q)

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
            .with_max_duration_micros(50000)\
            .with_max_outer_iterations(15)
        
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                meta,
                                                build_config,
                                                solver_config)
        builder.build()
       

if __name__=='__main__':
    mpc = MPCGenerator(nr_of_robots=5)
    mpc.build_mpc()