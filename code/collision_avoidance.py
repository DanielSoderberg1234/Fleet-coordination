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
        q = 10 # Cost of deviating from x and y reference
        qtheta = 1 # Cost of deviating from angle reference
        r = 0.1 # Cost for control action
        qN = 100 # Final cost of deviating from x and y reference
        qthetaN = 1000 # Final cost of deviating from angle reference
        qobs = 200 # Cost for being closer than r to the other robot
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

    def plot_robot(self,x,y,theta): 
        # Width of robot
        width = 0.5
        length = 0.7

        # Define rectangular shape of the robot
        corners = np.array([[length/2,width/2], 
                            [length/2,-width/2],
                            [-length/2,-width/2],
                            [-length/2,width/2],
                            [length/2,width/2]]).T
        
        # Define rotation matrix
        rot = np.array([[ np.cos(theta), -np.sin(theta)],[ np.sin(theta), np.cos(theta)]])

        # Rotate rectangle with the current angle
        rot_corners = rot@corners

        # Plot the robot with center x,y
        plt.plot(x+rot_corners[0,:], y+rot_corners[1,:],color='k')


    def plot_again(self,x1,y1,theta1,x2,y2,theta2): 
        # Append to the actual path taken
        self.past_traj1x.append(x1[0])
        self.past_traj1y.append(y1[0])

        # Append to the actual path taken
        self.past_traj2x.append(x2[0])
        self.past_traj2y.append(y2[0])

        plt.cla()
        ang = np.linspace(0,2*np.pi,100)
        r=1.0

        plt.plot(self.past_traj1x,self.past_traj1y,'-o',color='r',label="Actual1")
        plt.plot(x1[1:],y1[1:],'-o',color='r',alpha=0.2, label="Predicted1")
        plt.plot(x1[0]+r*np.cos(ang), y1[0]+r*np.sin(ang),'-',color='k')
        self.plot_robot(x1[0],y1[0],theta1)

        plt.plot(self.past_traj2x,self.past_traj2y,'-o',color='b',label="Actual2")
        plt.plot(x2[1:],y2[1:],'-o',color='b',alpha=0.2, label="Predicted2")
        plt.plot(x2[0]+r*np.cos(ang), y2[0]+r*np.sin(ang),'-',color='k')
        self.plot_robot(x2[0],y2[0],theta2)

        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.legend()
        plt.pause(0.1)

    def update_reference_trajectory(self,traj,rest):
        # Modify the reference trajectory
        if len(rest) == 0: 
            # If we have no more points to append to our trajectory, 
            # Just add the last point
            traj.extend(traj[-3:])
        else: 
            traj.extend(rest[:3])
            rest = rest[3:]

        return traj, rest

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
        self.plot_again(xlist1,ylist1,theta1,xlist2,ylist2,theta2)

        # Remove first state point and continue
        [traj1.pop(0) for i in range(0,3)]
        [traj2.pop(0) for i in range(0,3)]
        
        # The state we are at is given by applying the first control input
        x1,y1,theta1 = model(x1,y1,theta1,u1[:self.nu],self.ts)
        x2,y2,theta2 = model(x2,y2,theta2,u2[:self.nu],self.ts)

        traj1[:3] = [x1,y1,theta1]
        traj2[:3] = [x2,y2,theta2]

        return traj1, traj2

    def run(self, traj1, traj2): 
        # Make sure that the plots are non-blocking
        plt.show(block=False)

        rest1 = traj1[(self.N+1)*self.nx:]
        rest2 = traj2[(self.N+1)*self.nx:]

        traj1 = traj1[:(self.N+1)*self.nx]
        traj2 = traj2[:(self.N+1)*self.nx]
        
        # Run from the next step
        for j in range(0,40+1):    
            # Run collision avoidance again
            traj1, traj2 = self.run_one_iteration(traj1, traj2)
            traj1, rest1= self.update_reference_trajectory(traj1, rest1)
            traj2, rest2 = self.update_reference_trajectory(traj2, rest2)

        plt.pause(2)
        plt.close()

if __name__=="__main__": 
    avoid = CollisionAvoidance()

    # Case 1 - Crossing
    #traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    #traj2 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    #avoid.run(traj1, traj2)

    # Case 2 - Towards eachother
    #traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    #traj2 = generate_straight_trajectory(x=2,y=0,theta=-cs.pi,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    #avoid.run(traj1, traj2)

    # Case 3 - Behind eachother
    traj1 = generate_straight_trajectory(x=-1,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=-2.1,y=0,theta=0,v=1.3,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    avoid.run(traj1, traj2)

    avoid.mng.kill()