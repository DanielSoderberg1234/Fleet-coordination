from time import process_time
import matplotlib.pyplot as plt
import warnings
import opengen as og
from function_lib import model
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import ConvexHull
#import cdd
from function_lib import compute_polytope_halfspaces
import casadi.casadi as cs
from itertools import combinations
from scipy.spatial import Delaunay

def shortest_dist_to_ref(ref,x,y,N):
    """
        Calculations according to: 
            https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment

        A line is defined from two points s1 and s2 as: 
            s(t) = s1 + t*(s2-s1)

        Distance from a point p to the line is given by:
            d(t) = ||s(t)-p||_2

        Take derive and set it to zero to find optimal value of t: 
            t_hat = <p-s1,s2-s1> / ||s2-s1||_2

        Project to the line 
            t_star = min(max(t_hat,0),1)

        Closest point is now: 
            st = s1 + t_star*(s2-s1)

        Vector pointing from the point to the line: 
            dt = st-p

        Distance from the point to the line 
            dist = dt(0)**2 + dt(1)**2
    """
    # Extract the references for readability
    x_ref = ref[0::5]
    y_ref = ref[1::5]

    # Define the point we are at
    p = cs.vertcat(x,y)
    
    # Get fist point for the line segment
    s1 = cs.vertcat(x_ref[0],y_ref[0])
    
    # Variable for holding the distance to each line
    dist_vec = cs.SX.ones(1)

    # Loop over all possible line segments
    for i in range(1,N):
        # Get the next point
        s2 = cs.vertcat(x_ref[i],y_ref[i])

        #print("\nCurrent linesegment:  {}  <=>  {}  ".format(s1,s2))

        # Calculate t_hat and t_star
        t_hat = cs.dot(p-s1,s2-s1)/((s2[1]-s1[1])**2 + (s2[0]-s1[0])**2 + 1e-16)
        
        t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0)
        
        # Get the closest point from the line s
        st = s1 + t_star*(s2-s1)
        #print("Closes point to:  {}  <=>  {}".format(p,st))
        # Vector from point to the closest point on the line
        dvec = st-p
        
        # Calculate distance
        dist = dvec[0]**2+dvec[1]**2
        #print("The distance is:  {} ".format(dist))
        # Add to distance vector 
        dist_vec = cs.horzcat(dist_vec, dist)

        # Update s1
        s1 = s2
     

    return cs.mmin(dist_vec[1:])

def cost_line(ref,x,y,N, qdist): 
    shortest_dist = shortest_dist_to_ref(ref,x,y,N)
    return shortest_dist*qdist

def cost_control_action(u,u_ref,rv,rw): 
    cost = rv*(u_ref[0] - u[0])**2
    cost += rw*(u_ref[1] - u[1])**2
    return cost 

def bound_control_action(vmin,vmax,wmin,wmax,N): 
    # But hard constraints on the velocities of the robot
    umin = [vmin,wmin]*N
    umax = [vmax,wmax]*N
    return og.constraints.Rectangle(umin, umax)

# Reference vector 
(nx,nu,N,ts) = (5,2,20,0.1)

# Control action 
u = cs.SX.sym('u',N*nu)

# Reference 
p = cs.SX.sym('p', (N+1)*nx )

# Weights 
Q = cs.SX.sym('Q', 3)
qdist, rv, rw = Q[0], Q[1], Q[2]

# Starting state
x,y,theta = p[0],p[1],p[2]

# Cost 
cost = 0 

for i,j in zip(range(0,nx*N,nx),range(0,nu*N,nu)):
    # Get the control action for this step  
    ui = u[j:j+nu]

    # Get the reference for this step 
    refi = p[i:i+nx]
    u_refi = refi[3:]

    # Cost for control action
    cost += cost_control_action(ui,u_refi,rv,rw)

    # Update state 
    x,y,theta = model(x,y,theta,ui,ts)

    # Cost for deviating from the 
    cost += cost_line(p,x,y,N+1,qdist)

bounds = bound_control_action(vmin=0.0,vmax=1.5,wmin=-1,wmax=1,N=N)
p = cs.vertcat(p,Q)

problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

build_config = og.config.BuildConfiguration()\
    .with_build_directory("nmpc_cte")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta()\
    .with_optimizer_name("test_cte")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)\
    .with_max_duration_micros(50000)\
    .with_max_outer_iterations(15)

builder = og.builder.OpEnOptimizerBuilder(problem,
                                        meta,
                                        build_config,
                                        solver_config)
builder.build()