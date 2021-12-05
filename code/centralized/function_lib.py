import casadi.casadi as cs
import numpy as np
from scipy.spatial import ConvexHull
import cdd
from shapely.geometry import Polygon

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

def generate_turn_right_trajectory(x,y,theta,v,ts,N1,N2): 
    states = [x,y,theta,v,0]

    theta_turn = theta-cs.pi/2
    if theta_turn < 0:
        theta_turn+2*cs.pi

    for i in range(0,N1): 
        x,y,theta = model(x,y,theta,[v,0],ts=ts)
        states.extend([x,y,theta,v,0])

    for i in range(0,N2): 
        if i == N2-1: 
            v = 0
        x,y,theta_turn = model(x,y,theta_turn,[v,0],ts=ts)
        states.extend([x,y,theta_turn,v,0])

    return states

def control_action_to_trajectory(x,y,theta,u,ts): 
        # Get the linear and angular velocities
        v = u[0::2]
        w = u[1::2]

        # Create a list of x and y states
        xlist = []
        ylist = []

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],ts)
            xlist.append(x)
            ylist.append(y)

        return xlist,ylist

def predict(x,y,dx,dy, N, ts):
        xy = []
        for i in range(0,N): 
                x += ts*dx
                y += ts*dy
                xy.append(x)
                xy.append(y)
        return xy


def unpadded_square(xc,yc,width,height): 
    return Polygon( [[xc-width/2, yc-height/2],[xc+width/2, yc-height/2],[xc+width/2, yc+height/2],[xc-width/2, yc+height/2] ])

def padded_square(xc,yc,width,height, pad): 
    padw = (pad+width/2)
    padh = (pad+height/2)
    return Polygon( [[xc-padw, yc-padh],[xc+padw, yc-padh],[xc+padw, yc+padh],[xc-padw, yc+padh] ])


def polygon_to_eqs(polygon): 
        if polygon == None: 
            return [0]*12
        vertices = polygon.exterior.coords[:-1]
        A, b = compute_polytope_halfspaces(vertices)
        return [A[0,0],A[0,1],b[0],A[1,0],A[1,1],b[1],A[2,0],A[2,1],b[2],A[3,0],A[3,1],b[3] ]

def compute_polytope_halfspaces(vertices):
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
    return (A, b)