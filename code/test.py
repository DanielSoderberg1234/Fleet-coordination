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

def polygon_to_constraint(vertices): 
    # Triangulate the vertices, returns the index of corners for that triangle
    tr = Delaunay(vertices)
    # Get the traingles defined by the vertices
    tri = vertices[tr.simplices]
    # Array of the equation
    eqs = []
    for t in tri: 
        # Compute the halfspaces
        A, b = compute_polytope_halfspaces(t)
        # Put the values into the desired format
        eqs.extend([A[0,0],A[0,1],b[0],A[1,0],A[1,1],b[1],A[2,0],A[2,1],b[2]])
    
    plt.plot(tri[0,:,0],tri[0,:,1])
    plt.plot(tri[1,:,0],tri[1,:,1])
    plt.plot(.1,.1,'x')
    plt.show()
    return eqs

def cost_inside_polygon(robots,b): 
        cost = 0.0
        for robot_id in robots: 
            x,y,theta = robots[robot_id]['State']

            nr_of_traingles = 2

            nr_of_params = 9

           
            for i in range(0,nr_of_traingles): 
                params = b[i*nr_of_params:(i+1)*nr_of_params]

                inside = 1
                for j in range(0,nr_of_params,3):
                    h = params[j:j+3]
                    inside *= cs.fmax(0.0, h[2] - h[1]*y - h[0]*x )

                print(inside)

                cost += 100*inside


        return cost



vertices = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
eqs = polygon_to_constraint(vertices)
robots = {}
robots[0] = {'State': [.1,.1,0]}
cost = cost_inside_polygon(robots, eqs)
print(cost)



