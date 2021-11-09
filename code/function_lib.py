import casadi.casadi as cs
import numpy as np
from scipy.spatial import ConvexHull
#import cdd

def model(x,y,theta,u,ts):
    # Get the velocities for readability 
    v,w = u[0], u[1]

    # Update according to the model
    x += ts*cs.cos(theta)*v
    y += ts*cs.sin(theta)*v
    theta += ts*w

    return x,y,theta

def generate_straight_trajectory(x,y,theta,v,ts,N): 
    states = [x,y,theta]

    for i in range(0,N): 
        x,y,theta = model(x,y,theta,[v,0],ts=ts)
        states.extend([x,y,theta])

    return states

#def compute_polytope_halfspaces(vertices):
    """

    Implementation from: https://github.com/stephane-caron/pypoman
    
    Compute the halfspace representation (H-rep) of a polytope defined as
    convex hull of a set of vertices:
    .. math::
        A x \\leq b
        \\quad \\Leftrightarrow \\quad
        x \\in \\mathrm{conv}(\\mathrm{vertices})
    Parameters
    ----------
    vertices : list of arrays
        List of polytope vertices.
    Returns
    -------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
    """
"""
    V = np.vstack(vertices)
    t = np.ones((V.shape[0], 1))  # first column is 1 for vertices
    tV = np.hstack([t, V])
    mat = cdd.Matrix(tV, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(mat)
    bA = np.array(P.get_inequalities())
    if bA.shape == (0,):  # bA == []
        return bA
    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])
    return (A, b)"""