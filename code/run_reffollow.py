import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory
import opengen as og

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
states1 = generate_straight_trajectory(-1,0,0,1,0.1,20) # Trajectory from x=-1, y=0 driving straight to the right
states2 = generate_straight_trajectory(1,0,-cs.pi,1,0.1,20) # Trajectory from x=0,y=-1 driving straight up
states3 = generate_straight_trajectory(0,-1,cs.pi/2,1,0.1,20) # Trajectory from x=0,y=-1 driving straight up

# Create the input vector
p = []
p.extend(states1)
p.extend(states2)
p.extend(states3)
p.extend(weights)

# Call the solver
mng = og.tcp.OptimizerTcpManager('reffollow/version2')
mng.start()
mng.ping()
solution = mng.call(p, initial_guess=[1.0] * (3*nu*N))
mng.kill()

# Extract the sultion 
u_star = solution['solution']


# Get the generated trajectory for the first robot
u1 = u_star[:nu*N]
x1,y1,theta1 = -1,0,0
ts = 0.1
u2 = u_star[nu*N:2*nu*N]
x2,y2,theta2 = 1,0,-cs.pi
u3 = u_star[2*nu*N:]
x3,y3,theta3 = 0,-1,cs.pi/2

x1, y1 = control_action_to_trajectory(x1,y1,theta1,u1,ts)
x2, y2 = control_action_to_trajectory(x2,y2,theta2,u2,ts)
x3, y3 = control_action_to_trajectory(x3,y3,theta3,u3,ts)


past_traj1x, past_traj1y = [],[]
past_traj2x, past_traj2y = [],[]
past_traj3x, past_traj3y = [],[]

plt.show(block=False)
for i in range(0,21): 
    past_traj1x.append(x1[0])
    past_traj1y.append(y1[0])
    past_traj2x.append(x2[0])
    past_traj2y.append(y2[0])
    past_traj3x.append(x3[0])
    past_traj3y.append(y3[0])


    plt.cla()
    ang = np.linspace(0,2*np.pi,100)
    r=0.25
    plt.plot(past_traj1x,past_traj1y,'-o',color='r',label="Actual1")
    plt.plot(x1[1:],y1[1:],'-o',color='r',alpha=0.2, label="Predicted1")
    #plt.plot(x1[0]+r*np.cos(ang), y1[0]+r*np.sin(ang),color='k')

    plt.plot(past_traj2x,past_traj2y,'-o',color='b',label="Actual2")
    plt.plot(x2[1:],y2[1:],'-o',color='b',alpha=0.2, label="Predicted2")
    plt.plot(x2[0]+r*np.cos(ang), y2[0]+r*np.sin(ang),color='k')

    plt.plot(past_traj3x,past_traj3y,'-o',color='g',label="Actual3")
    plt.plot(x3[1:],y3[1:],'-o',color='g',alpha=0.2, label="Predicted3")
    plt.plot(x3[0]+r*np.cos(ang), y3[0]+r*np.sin(ang),color='k')

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.legend()
    plt.pause(0.5)

    x1 = x1[1:]
    y1 = y1[1:]
    x2 = x2[1:]
    y2 = y2[1:]
    x3 = x3[1:]
    y3 = y3[1:]



