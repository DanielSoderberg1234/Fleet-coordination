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

def plot_again(x1,y1,x2,y2, past_traj1x, past_traj1y, past_traj2x,past_traj2y): 
        plt.cla()
        ang = np.linspace(0,2*np.pi,100)
        r=0.25

        plt.plot(past_traj1x,past_traj1y,'-o',color='r',label="Actual1")
        plt.plot(x1[1:],y1[1:],'-o',color='r',alpha=0.2, label="Predicted1")
        plt.plot(x1[0]+r*np.cos(ang), y1[0]+r*np.sin(ang),color='k')

        plt.plot(past_traj2x,past_traj2y,'-o',color='b',label="Actual2")
        plt.plot(x2[1:],y2[1:],'-o',color='b',alpha=0.2, label="Predicted2")
        plt.plot(x2[0]+r*np.cos(ang), y2[0]+r*np.sin(ang),color='k')

        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.legend()

# Some parameters
nu = 2
N = 20

# Define weights and and get the reference trajectories
weights = [0.1,100,0.01,0.1,100,100]
states1 = generate_straight_trajectory(-1,0,0,1,0.1,20) # Trajectory from x=-1, y=0 driving straight to the right
states2 = generate_straight_trajectory(0,-1,cs.pi/2,1,0.1,20) # Trajectory from x=0,y=-1 driving straight up

# Create the input vector
p = []
p.extend(states1)
p.extend(states2)
p.extend(weights)

# Call the solver
mng = og.tcp.OptimizerTcpManager('reffollow/version1')
mng.start()
mng.ping()
solution = mng.call(p, initial_guess=[1.0] * (2*nu*N))
mng.kill()

# Extract the sultion 
u_star = solution['solution']


# Get the generated trajectory for the first robot
u1 = u_star[:nu*N]
x1,y1,theta1 = -1,0,0
ts = 0.1
u2 = u_star[nu*N:]
x2,y2,theta2 = 0,-1,cs.pi/2

xlist1, ylist1 = control_action_to_trajectory(x1,y1,theta1,u1,ts)
xlist2, ylist2 = control_action_to_trajectory(x2,y2,theta2,u2,ts)

past_traj1x, past_traj1y = [],[]
past_traj2x, past_traj2y = [],[]

plt.show(block=False)
for i in range(0,21): 
    past_traj1x.append(xlist1[0])
    past_traj1y.append(ylist1[0])
    past_traj2x.append(xlist2[0])
    past_traj2y.append(ylist2[0])
    plot_again(xlist1,ylist1,xlist2,ylist2,past_traj1x,past_traj1y,past_traj2x, past_traj2y)
    plt.pause(0.5)
    xlist1 = xlist1[1:]
    ylist1 = ylist1[1:]
    xlist2 = xlist2[1:]
    ylist2 = ylist2[1:]



