import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory
import opengen as og
import warnings
warnings.filterwarnings("ignore")

"""
 A file for testing the MPC. 
 Calls the MPC and modifies the trajectories multiple times.
 It runs the robots with the assumption that they 
 follow the trajectory exactly. A new MPC call is done at every times instance
 It plots the predicted path, where it's at and a 0.25 m distance constrain.
"""


class CollisionAvoidance: 
    def __init__(self):
        # Some parameters 
        self.nx = 3 # Number of states for each robot
        self.nu = 2 # Nr of control inputs
        self.N = 20 # Length of horizon 
        self.ts = 0.1 # Sampling time
        q = 0.1
        qtheta = 1
        r = 0.1
        qN = 0.1
        qthetaN = 1
        qobs = 1000000
        self.weights = [q,qtheta,r,qN,qthetaN,qobs]

        # Create the solver and open a tcp port to it 
        self.mng = og.tcp.OptimizerTcpManager('reffollow/version1')
        self.mng.start()
        self.mng.ping()

        # Past trajectories
        self.past_traj1x = []
        self.past_traj1y = []

        self.past_traj2x = []
        self.past_traj2y = []
        

    def get_input(self, traj1, traj2):
        # Create the input vector
        p = []
        p.extend(traj1)
        p.extend(traj2)
        p.extend(self.weights)
        return p

    def control_action_to_trajectory(self,x,y,theta,u): 
        # Get the linear and angular velocities
        v = u[0::2]
        w = u[1::2]

        # Create a list of x and y states
        xlist = [x]
        ylist = [y]

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],self.ts)
            xlist.append(x)
            ylist.append(y)

        return xlist,ylist

    def plot_again(self,x1,y1,x2,y2): 
        # Append to the actual path taken
        self.past_traj1x.append(x1[0])
        self.past_traj1y.append(y1[0])

        # Append to the actual path taken
        self.past_traj2x.append(x2[0])
        self.past_traj2y.append(y2[0])

        plt.cla()
        ang = np.linspace(0,2*np.pi,100)
        r=0.25

        plt.plot(self.past_traj1x,self.past_traj1y,'-o',color='r',label="Actual1")
        plt.plot(x1[1:],y1[1:],'-o',color='r',alpha=0.2, label="Predicted1")
        plt.plot(x1[0]+r*np.cos(ang), y1[0]+r*np.sin(ang),color='k')

        plt.plot(self.past_traj2x,self.past_traj2y,'-o',color='b',label="Actual2")
        plt.plot(x2[1:],y2[1:],'-o',color='b',alpha=0.2, label="Predicted2")
        plt.plot(x2[0]+r*np.cos(ang), y2[0]+r*np.sin(ang),color='k')

        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.legend()
        

    def run_one_iteration(self, traj1,traj2): 
        # Get the input
        mpc_input = self.get_input(traj1,traj2)

        # Call the solver
        solution = self.mng.call(p=mpc_input, initial_guess=[1.0] * (2*self.nu*self.N))

        # Get the solver output 
        u_star = solution['solution']
        u1 = u_star[:self.nu*self.N]
        u2 = u_star[self.nu*self.N:]

        # Initial states
        x1,y1,theta1 = traj1[0],traj1[1], traj1[2]
        x2,y2,theta2 = traj2[0],traj2[1], traj2[2]

        # Get the trajectories
        xlist1, ylist1 = self.control_action_to_trajectory(x1,y1,theta1,u1)
        xlist2, ylist2 = self.control_action_to_trajectory(x2,y2,theta2,u2)

        # Plot the trajectories   
        self.plot_again(xlist1,ylist1,xlist2,ylist2)
        plt.pause(0.5)

    def run(self): 
        # Make sure that the plots are non-blocking
        plt.show(block=False)
        # Desired trajectories
        traj1 = generate_straight_trajectory(-1,0,0,1,0.1) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(0,-1,cs.pi/2,1,0.1) # Trajectory from x=0,y=-1 driving straight up

        # Run for inital trajctories
        self.run_one_iteration(traj1, traj2)

        # Run from the next step
        for j in range(0,self.N-1):    
            # Remove first state point and continue
            [traj1.pop(0) for i in range(0,3)]
            [traj2.pop(0) for i in range(0,3)]
            
            # Append last state to make sure that N = 20
            [traj1.append(traj1[-3]) for i in range(0,3)]
            [traj2.append(traj2[-3]) for i in range(0,3)]
        
            # Run collision avoidance again
            self.run_one_iteration(traj1, traj2)

        self.mng.kill()

if __name__=="__main__": 
    avoid = CollisionAvoidance()
    avoid.run()