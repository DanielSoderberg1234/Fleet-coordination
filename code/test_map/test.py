from time import process_time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import ConvexHull
#import cdd
from function_lib import compute_polytope_halfspaces
import casadi.casadi as cs
from itertools import combinations
from scipy.spatial import Delaunay

def closest_point(ref,x,y,N):
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

        print("\nCurrent linesegment:  {}  <=>  {}  ".format(s1,s2))

        # Calculate t_hat and t_star
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
        # Add to distance vector 
        dist_vec = cs.horzcat(dist_vec, dist)

        # Update s1
        s1 = s2
     

    return cs.mmin(dist_vec[1:])

# Ref vector is x,y,theta,v,w,...
ref = [1,1,0,0,0,2,2,0,0,0,3,3,0,0,0]
x = 1.5
y = 2
N = 3
#closest_point(ref,x,y,N)

print(ref[3:5])



