import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory, compute_polytope_halfspaces
import opengen as og
from scipy.spatial import Delaunay



def boundary_to_constraint(vertices): 
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
    
    return eqs


"""
 A file for testing the MPC. 
 Calls the MPC and modifies the trajectories once
 then it runs the robots with the assumption that they 
 follow exactly. It plots the predicted path, where it is
 at and a 0.25 m distance constrain.
"""
def control_action_to_trajectory(x,y,theta,u,ts): 
        # Get the linear and angular velocities
        v = u[0::2]
        w = u[1::2]

        # Create a list of x and y states
        xlist = [x]
        ylist = [y]

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],ts)
            xlist.append(x)
            ylist.append(y)

        return xlist,ylist

# Some parameters
nu = 2
N = 20

# Define weights and and get the reference trajectories
weights = [10,1,0.01,100,10,200]
states1 = generate_straight_trajectory(-.5,0,0,1,0.1,20) # Trajectory from x=-1, y=0 driving straight to the right
states2 = generate_straight_trajectory(0,-.5,cs.pi/2,1,0.1,20) # Trajectory from x=0,y=-1 driving straight up

vertices = np.array([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]])
eqs = boundary_to_constraint(vertices)

# Create the input vector
p = []
p.extend(states1)
p.extend(states2)
p.extend(weights)
p.extend(eqs)

# Call the solver
mng = og.tcp.OptimizerTcpManager('collision_avoidance/robot_2_solver')
mng.start()
mng.ping()
solution = mng.call(p, initial_guess=[1.0] * (2*nu*N))
mng.kill()

# Extract the sultion 
u_star = solution['solution']


# Get the generated trajectory for the first robot
u1 = u_star[:nu*N]
x1,y1,theta1 = -.5,0,0
ts = 0.1
u2 = u_star[nu*N:2*nu*N]
x2,y2,theta2 = 0,-.5,cs.pi/2


x1, y1 = control_action_to_trajectory(x1,y1,theta1,u1,ts)
x2, y2 = control_action_to_trajectory(x2,y2,theta2,u2,ts)



past_traj1x, past_traj1y = [],[]
past_traj2x, past_traj2y = [],[]


plt.show(block=False)
for i in range(0,21): 
    past_traj1x.append(x1[0])
    past_traj1y.append(y1[0])
    past_traj2x.append(x2[0])
    past_traj2y.append(y2[0])
    


    plt.cla()
    ang = np.linspace(0,2*np.pi,100)
    r=0.25
    plt.plot(past_traj1x,past_traj1y,'-o',color='r',label="Actual1")
    plt.plot(x1[1:],y1[1:],'-o',color='r',alpha=0.2, label="Predicted1")
    #plt.plot(x1[0]+r*np.cos(ang), y1[0]+r*np.sin(ang),color='k')

    plt.plot(past_traj2x,past_traj2y,'-o',color='b',label="Actual2")
    plt.plot(x2[1:],y2[1:],'-o',color='b',alpha=0.2, label="Predicted2")
    plt.plot(x2[0]+r*np.cos(ang), y2[0]+r*np.sin(ang),color='k')

    plt.plot([-.2,.2,.2,-.2,-.2], [-.2,-.2,.2,.2,-.2], color='g')

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.legend()
    plt.pause(0.5)

    x1 = x1[1:]
    y1 = y1[1:]
    x2 = x2[1:]
    y2 = y2[1:]




