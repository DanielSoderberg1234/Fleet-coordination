import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model

"""
    A file that generates the MPC formulation.
"""
class MPCGenerator: 
    def __init__(self): 
        self.name = "Fleet-collison"

    def cost_state_ref(self,x,y,theta,xref,yref,thetaref,q,qtheta): 
        # Cost for deviating from the current reference
        return q*( (xref-x)**2 + (yref-y)**2 ) + qtheta*(thetaref-theta)**2

    def cost_robot2robot_dist(self,x1,y1,x2,y2,qobs): 
        # Cost for being closer than r to the other robot
        return qobs*cs.fmax(0.0, 1**2 - (x1-x2)**2 - (y1-y2)**2)
    
    def cost_control_action(self,u,r): 
        # Cost for the control action
        return r*cs.dot(u, u)

    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        # But hard constraints on the velocities of the robot
        N = 2*N
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 3, 20, 0.1)

        # Input vector 2 trajectories, N long with nx states in each i=0,1,2,..,N-1 and the 6 last are the weights
        p = cs.SX.sym('p',2*nx*(N+1)+6)

        # Optimization variables 2 robots each with nu control inputs for N steps
        u = cs.SX.sym('u',2*nu*N)

        # First part of p is the trajectory reference for the first robot
        ref1 = p[:(N+1)*nx]

        # Second part of p is the trajectory reference for the second robot
        ref2 = p[(N+1)*nx:2*(N+1)*nx]

        # Define the structure of the optimization variables
        u1 = u[:nu*N]
        u2 = u[nu*N:]

        # Init states
        x1,y1,theta1 = p[0],p[1],p[2]
        x2,y2,theta2 = p[nx*(N+1)],p[nx*(N+1)+1],p[nx*(N+1)+2]

        # Get weights from input vectir as the last elements
        q, qtheta, r, qN, qthetaN,qobs = p[-6],p[-5],p[-4],p[-3],p[-2],p[-1]

        # Define the cost
        cost = 0
        
        # Just dummy so that they are defined
        xref1,yref1,thetaref1 = 0,0,0
        xref2,yref2,thetaref2 = 0,0,0

        for i,j in zip( range(0,nx*N,nx), range(0,nu*N,nu)): 
            # Extract the reference for that state
            ref1i = ref1[i:i+3]
            xref1, yref1, thetaref1 = ref1i[0], ref1i[1], ref1i[2]

            ref2i = ref2[i:i+3]
            xref2, yref2, thetaref2 = ref2i[0], ref2i[1], ref2i[2]
            
            # Cost for deviating from reference
            cost += self.cost_state_ref(x1,y1,theta1,xref1,yref1,thetaref1,q,qtheta)
            cost += self.cost_state_ref(x2,y2,theta2,xref2,yref2,thetaref2,q,qtheta)

            # Get current inputs
            u1i = u1[j:j+nu]
            u2i = u2[j:j+nu]

            # Cost for control action
            cost += self.cost_control_action(u1i,r)
            cost += self.cost_control_action(u2i,r)

            # Update states
            x1,y1,theta1 = model(x1,y1,theta1,u1i,ts)
            x2,y2,theta2 = model(x2,y2,theta2,u2i,ts)

            # Cost for being closer that a given distance to another robot
            cost += self.cost_robot2robot_dist(x1,y1,x2,y2,qobs)

        # Extract the last reference for that state
        ref1i = ref1[-3:]
        xref1, yref1, thetaref1 = ref1i[0], ref1i[1], ref1i[2]

        ref2i = ref2[-3:]
        xref2, yref2, thetaref2 = ref2i[0], ref2i[1], ref2i[2]

        # Final cost for deviating from state
        cost += self.cost_state_ref(x1,y1,theta1,xref1,yref1,thetaref1,qN,qthetaN)
        cost += self.cost_state_ref(x2,y2,theta2,xref2,yref2,thetaref2,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=-1.5,vmax=1.5,wmin=-1,wmax=1,N=N)

        return u,p,cost,bounds

    def build_mpc(self): 
        u,p,cost,bounds = self.generate_mpc_formulation()
        problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

        build_config = og.config.BuildConfiguration()\
            .with_build_directory("reffollow")\
            .with_build_mode("debug")\
            .with_tcp_interface_config()

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("version1")

        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-5)\
        
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                meta,
                                                build_config,
                                                solver_config)
        builder.build()


if __name__=='__main__':
    mpc = MPCGenerator()
    mpc.build_mpc()