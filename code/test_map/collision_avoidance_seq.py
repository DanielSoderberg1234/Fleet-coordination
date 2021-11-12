import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from RobotModelData import RobotModelData

"""
 A file for testing the MPC. 
 Calls the MPC and modifies the trajectories multiple times.
 It runs the robots with the assumption that they 
 follow the trajectory exactly. A new MPC call is done at every times instance
 It plots the predicted path, where it's at and a 0.25 m distance constrain.
"""


class CollisionAvoidance: 
    def __init__(self, r_model: RobotModelData):
        # Load parameters 
        self.nr_of_robots = r_model.nr_of_robots
        self.nx = r_model.nx
        self.nu = r_model.nu 
        self.N = r_model.N 
        self.ts = r_model.ts 
        self.weights = r_model.get_weights()

        self.pos_list = []

        # Create the solver and open a tcp port to it 
        self.mng = og.tcp.OptimizerTcpManager('collision_avoidance/robot_seq_solver')
        self.mng.start()
        self.mng.ping()

        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []


    def control_action_to_trajectory(self,x,y,theta,u): 
        # Get the linear and angular velocities
        v = u[0::2]
        w = u[1::2]

        # Create a list of x and y states
        xlist = []
        ylist = []

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],self.ts)
            xlist.append(x)
            ylist.append(y)

        return xlist,ylist

    def update_state(self, robot): 
        x,y,theta = robot['State']
        x,y,theta = model(x,y,theta,robot['u'][:2],self.ts)
        robot['State'] = [x,y,theta]

    def update_ref(self,robot):
        # Shift reference once step to the left
        robot['Ref'][:self.nx*(self.N-1)] = robot['Ref'][self.nx:]

        if len(robot['Remainder']) > 0:
            robot['Ref'][-self.nx:] = robot['Remainder'][:self.nx]
            del robot['Remainder'][:self.nx]

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
    
    def plot_safety_cricles(self, x,y): 
        ang = np.linspace(0,2*np.pi,100)
        r=1.0
        plt.plot(x+r*np.cos(ang), y+r*np.sin(ang),'-',color='k')

    
    # TODO: Kalla på denna funktion på lämpligt ställe
    def update_traj_list(self, robot):
        #Ta emot controllsignaler och lägg in pred pos i lista.
        x,y,theta = robot['State']

        # Calculate all fute x and y states
        x_pred, y_pred = self.control_action_to_trajectory(x,y,theta,robot['u'])
        
        self.pos_list = self.pos_list.append(x_pred, y_pred) 
        
        

    def plot_for_one_robot(self,robot):
        #TODO: Plotta listan med positions.

        x,y,theta = robot['State']

        # Calculate all fute x and y states
        x_pred, y_pred = self.control_action_to_trajectory(x,y,theta,robot['u'])

        # Save the states that we have been to
        robot['Past_x'].append(x)
        robot['Past_y'].append(y)

        plt.plot(robot['Past_x'],robot['Past_y'],'-o', color=robot['Color'])
        plt.plot(self.x_pred,self.y_pred,'-o', alpha=0.2,color=robot['Color'])
        self.plot_robot(x,y,theta)
        #self.plot_safety_cricles(x,y)

    def plot_dist(self,robots): 
        
        plt.subplot(1,2,2)
        plt.cla()
        for comb in combinations(range(0,self.nr_of_robots),2):
            x1,y1,theta1 = robots[comb[0]]['State']
            x2,y2,theta2 = robots[comb[1]]['State']   
            dist = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
            self.dist[comb].append(dist)
            plt.plot(self.dist[comb], label="Distance for {}".format(comb))
            lim_dist = 1
            plt.plot(range(0,len(self.dist[comb])), [lim_dist]*len(self.dist[comb]), label="Limit")

        
        plt.legend()
        plt.title("Distance")

    
    def get_input(self, robot):
        # Create the input vector to the mpsgenerator
        p = []
        p.extend(self.pos_list) 
        p.extend(robot['State'])
        p.extend(robot['Ref'])
        p.extend(self.weights)      
        return p

    def run_one_iteration(self,robots): 
        
        # Loop through and plan a traj for every robot in a sequencial order. 
        for robot in robots: 
            mpc_input = self.get_input(robots)

            # Call the solver
            solution = self.mng.call(p=mpc_input, initial_guess=[1.0] * (self.nr_of_robots*self.nu*self.N))

            # Get the solver output           
            robot['u'] = solution['solution'] 

            #Update predicted positions
            self.update_traj_list(self, robot)

        #Plot iteration
        plt.subplot(1,2,1)
        plt.cla()
        for robot_id in robots: 
            self.plot_for_one_robot(robots[robot_id])
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        self.plot_dist(robots)
        plt.pause(0.01)
        
        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])
        
        
    def run(self, robots):
        plt.show(block=False)
        for i in range(0,40+1): 
            self.run_one_iteration(robots)
        plt.pause(2)

        

if __name__=="__main__": 

    # Case 1 - Crossing
    r_model = RobotModelData(nr_of_robots=2)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    
    robots = {}
    robots[0] = {"State": traj1[:3], 'Ref': traj1[3:20*3+3], 'Remainder': traj1[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:3], 'Ref': traj2[3:20*3+3], 'Remainder': traj2[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'b'}
    
    avoid.run(robots)
    avoid.mng.kill()

    # Case 2 - Towards eachother
    r_model = RobotModelData(nr_of_robots=2)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=2,y=0,theta=-cs.pi,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    robots = {}
    robots[0] = {"State": traj1[:3], 'Ref': traj1[3:20*3+3], 'Remainder': traj1[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:3], 'Ref': traj2[3:20*3+3], 'Remainder': traj2[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'b'}
    avoid.run(robots)
    avoid.mng.kill()


    # Case 3 - Behind eachother
    r_model = RobotModelData(nr_of_robots=2)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-1,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=-2.1,y=0,theta=0,v=1.3,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    robots = {}
    robots[0] = {"State": traj1[:3], 'Ref': traj1[3:20*3+3], 'Remainder': traj1[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:3], 'Ref': traj2[3:20*3+3], 'Remainder': traj2[20*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'b'}
    avoid.run(robots)
    avoid.mng.kill()
    """

    # Case 4 - 4 robots
    avoid = CollisionAvoidance(nr_of_robots=4)
    traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=60) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=3,y=0,theta=-cs.pi,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    traj3 = generate_straight_trajectory(x=1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    traj4 = generate_straight_trajectory(x=-1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    
    N = 20
    robots = {}
    robots[0] = {"State": traj1[:3], 'Ref': traj1[3:N*3+3], 'Remainder': traj1[N*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:3], 'Ref': traj2[3:N*3+3], 'Remainder': traj2[N*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'b'}
    robots[2] = {"State": traj3[:3], 'Ref': traj3[3:N*3+3], 'Remainder': traj3[N*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'g'}
    robots[3] = {"State": traj4[:3], 'Ref': traj4[3:N*3+3], 'Remainder': traj4[N*3+3:], 'u': [], 'Past_x': [], 'Past_y': [], 'Color': 'm'}

    avoid.run(robots)
    avoid.mng.kill()
    """

    """
    # How to change all parameters in r_model
    r_model = RobotModelData(nr_of_robots = 2,
                        nx = 3, 
                        nu = 2, 
                        N = 20, 
                        ts = 0.1, 
                        q = 10, 
                        qtheta = 1,
                        r = 0.01, 
                        qN = 100, 
                        qthetaN = 10, 
                        qobs = 200, 
                        )
    """

    