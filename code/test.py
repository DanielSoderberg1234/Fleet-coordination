import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import ConvexHull
#import cdd
from function_lib import compute_polytope_halfspaces
import casadi.casadi as cs
from itertools import combinations

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
dist = {}

i = 1
for comb in combinations([0,1,2],2): 
    dist[comb] = []
    i+=1

for comb in combinations([0,1,2],2): 
    print(dist[comb])