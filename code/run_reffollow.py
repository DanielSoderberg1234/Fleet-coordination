import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory
import opengen as og


# Some parameters
nu = 2
N = 20

# Define weights and and get the reference trajectories
weights = [0.1,100,0.01,0.1,100,100]
states1 = generate_straight_trajectory(-1,0,0,1,0.1) # Trajectory from x=-1, y=0 driving straight to the right
states2 = generate_straight_trajectory(0,-1,cs.pi/2,1,0.1) # Trajectory from x=0,y=-1 driving straight up

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
v1 = u1[0::2]
w1 = u1[1::2]


x1,y1,theta1 = -1,0,0
ts = 0.1

xlist1 = []
ylist1 = []

xlist1.append(x1)
ylist1.append(y1)

for vi,wi in zip(v1,w1): 
    x1,y1,theta1 = model(x1,y1,theta1,[vi,wi],ts)
    xlist1.append(x1)
    ylist1.append(y1)


# Get the generated trajectory for the second robot
u2 = u_star[nu*N:]
v2 = u2[0::2]
w2 = u2[1::2]

x2,y2,theta2 = 0,-1,cs.pi/2
ts = 0.1

xlist2 = []
ylist2 = []

xlist2.append(x2)
ylist2.append(y2)

for vi,wi in zip(v2,w2): 
    x2,y2,theta2 = model(x2,y2,theta2,[vi,wi],ts)
    xlist2.append(x2)
    ylist2.append(y2)

# Plot the solutions stationary
ref1x = states1[0::3]
ref1y = states1[1::3]
ref2x = states2[0::3]
ref2y = states2[1::3]
plt.subplot(1,2,1)
plt.plot(ref1x,ref1y,'-o',label="Reference1")
plt.plot(ref2x,ref2y,'-o',label="Reference2")
plt.plot(xlist1,ylist1,'-o', label="Generated1")
plt.plot(xlist2,ylist2,'-o', label="Generated2")

# Plot the sultions interactively
plt.subplot(1,2,2)
for i in range(0,21):
    #plt.pause(5)
    plt.plot(xlist1[:i],ylist1[:i],'-o')
    plt.plot(xlist2[:i],ylist2[:i],'-o')
    plt.pause(1)


print(len(xlist1))
plt.legend()
plt.show()
