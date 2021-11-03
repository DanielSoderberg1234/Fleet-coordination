import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model

class MPCBuilder: 
    def __init__(self): 
        self.name = "Fleet-collison"

    def cost_state_ref(self,x,y,theta,xref,yref,thetaref,q,qtheta): 
        return q*( (xref-x)**2 + (yref-y)**2 ) + qtheta*(thetaref-theta)**2

    def cost_robot2robot_dist(self,x,y,qobs): 
        return qobs*cs.fmax(0.0, 0.25**2 - x**2 - y**2)
    
    def cost_control_action(self,u,r): 
        return r*cs.dot(u, u)

    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)

    def generate_mpc_formulation(self): 

        # Some states
        (nu, nx, N, ts) = (2, 3, 20, 0.1)

        xref, yref, thetaref = 1,0,0

        # Optimization varibales 
        u = cs.SX.sym('u', nu*N)
        p = cs.SX.sym('p', nx+6)

        # Init states
        x,y,theta = p[0],p[1],p[2]
        q, qtheta, r, qN, qthetaN,qobs = p[3],p[4],p[5],p[6],p[7],p[8]

        cost = 0
        c = 0
        for i in range(0,nu*N,nu):
            # Cost for deviating from reference
            cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,q,qtheta)

            # Get current inputs
            ui = u[i:i+nu]

            # Cost for control action
            cost += self.cost_control_action(ui,r)

            # Update states
            x,y,theta = model(x,y,theta,ui,ts)

            # Cost for being closer that a given distance to another robot
            cost += self.cost_robot2robot_dist(x,y,qobs)

        # Final cost for deviating from state
        cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=-1,vmax=1,wmin=-1,wmax=1,N=N)

        return u,p,cost,bounds

    def build_mpc(self): 
        u,p,cost,bounds = self.generate_mpc_formulation()
        problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

        build_config = og.config.BuildConfiguration()\
            .with_build_directory("collision")\
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
    mpc = MPCBuilder()
    mpc.build_mpc()
