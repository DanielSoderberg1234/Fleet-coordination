import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import ConvexHull
import cdd
from function_lib import compute_polytope_halfspaces
import casadi.casadi as cs

"""
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
A, b = compute_polytope_halfspaces(vertices)
point = np.array([1.1,1.1])

inside_pol = 1
for i in range(0, A.shape[0]):
    inside_pol *= cs.fmax(0.0, b[i] - A[i,:]@point) 


outside_pol = 0
for i in range(0, A.shape[0]):
    outside_pol += cs.fmin(0.0, b[i] - A[i,:]@point)**2

print(outside_pol)

a = [1,2,3,4,5]
print(a[-3:])
"""
ang = np.pi/4

x = 1
y = 1
width = 0.5
length = 0.7

corners = np.array([[length/2,width/2], 
                    [length/2,-width/2],
                    [-length/2,-width/2],
                    [-length/2,width/2],
                    [length/2,width/2]]).T

rot = np.array([[ np.cos(ang), -np.sin(ang)],[ np.sin(ang), np.cos(ang)]])

rot_corners = rot@corners

plt.plot(x+corners[0,:], y+corners[1,:])
plt.plot(x+rot_corners[0,:], y+rot_corners[1,:])
plt.show()