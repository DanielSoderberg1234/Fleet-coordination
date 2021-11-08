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
u = cs.SX.sym('u',10)
p = cs.SX.sym('p',10)

states = [0,1,2]

variables = {}
variables[0] = [states,u[:5],p[:5]]
variables[1] = [states,u[5:],p[5:]]

for i in variables:
    print(variables[i])