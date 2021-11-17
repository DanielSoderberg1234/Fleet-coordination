from time import process_time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import ConvexHull
#import cdd
#from function_lib import compute_polytope_halfspaces
import casadi.casadi as cs
from itertools import combinations
from scipy.spatial import Delaunay

def model(x,y,theta,u,ts):
    # Get the velocities for readability 
    v,w = u[0], u[1]

    # Update according to the model
    x += ts*cs.cos(theta)*v
    y += ts*cs.sin(theta)*v
    theta += ts*w

    return x,y,theta

def generate_straight_trajectory(x,y,theta,v,ts,N): 
    states = [x,y,theta,v,0]

    for i in range(0,N): 
        if i == N-1: 
            v = 0
        x,y,theta = model(x,y,theta,[v,0],ts=ts)
        states.extend([x,y,theta,v,0])

    return states

def closest_point_and_ang(s1,s2,p):

    print("\nCurrent linesegment:  {}  <=>  {}  ".format(s1,s2))

    # Calculate t_hat and t_star
    ps1 = p-s1
    ps2 = p-s2

    
    t_hat = cs.dot(p-s1,s2-s1)/((s2[1]-s1[1])**2 + (s2[0]-s1[0])**2 + 1e-16)
    

    t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0)
    
    # Get the closest point from the line s
    st = s1 + t_star*(s2-s1)
    print("Closes point to:  {}  <=>  {}".format(p,st))
    # Vector from point to the closest point on the line
    dvec = st-p
    
    # Calculate distance
    dist = dvec[0]**2+dvec[1]**2
    print("The distance is:  {} ".format(dist))


    # Calculate angle
    ang = cs.acos( cs.dot(s1,s2)/( cs.norm_2(s2)*cs.norm_2(s1)) )
    
     

    return st, ang


s1 = cs.vertcat(1,0)
s2 = cs.vertcat(5,0)
p  = cs.vertcat(2,2)

st, ang = closest_point_and_ang(s1,s2,p)









