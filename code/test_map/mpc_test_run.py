import opengen as og
from function_lib import model, generate_straight_trajectory
import matplotlib.pyplot as plt


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

nx,nu,N,ts = 5,2,20,0.1

x,y,theta = -1.0,0,0
states1 = generate_straight_trajectory(x,y,theta,1,0.1,20)
weights = [10,1,1]

p = []
p.extend(states1)
p.extend(weights)

# Call the solver
mng = og.tcp.OptimizerTcpManager('nmpc_cte/test_cte')
mng.start()
mng.ping()
solution = mng.call(p, initial_guess=[1.0] * nu*N)
mng.kill()

u_star = solution['solution']
v = u_star[0::2]
w = u_star[1::2]

xlist,ylist = control_action_to_trajectory(x,y,theta,u_star,ts)
plt.subplot(1,2,1)
plt.plot(xlist,ylist)
plt.xlim(-2,2)
plt.ylim(-1,1)

plt.subplot(2,2,2)
plt.plot(v)
plt.ylim(0,2.0)

plt.subplot(2,2,4)
plt.plot(w)
plt.ylim(-1.5,1.5)

plt.show()