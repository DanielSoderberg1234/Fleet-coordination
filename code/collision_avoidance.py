import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from numpy.lib.arraypad import pad
from function_lib import model, generate_straight_trajectory
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
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


        # Create the solver and open a tcp port to it 
        self.mng = og.tcp.OptimizerTcpManager('collision_avoidance/robot_{}_solver'.format(self.nr_of_robots))
        self.mng.start()
        self.mng.ping()

        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []

        # Time 
        self.time = 0

        self.plots = {"Map": plt.subplot(1,2,1), "Velocity": plt.subplot(2,2,2), "Distance": plt.subplot(2,2,4) }

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
        x,y,theta,v,w = robot['State']
        robot['Past_v'].append(v)
        robot['Past_w'].append(w)
        x,y,theta = model(x,y,theta,robot['u'][:2],self.ts)
        robot['State'] = [x,y,theta,robot['u'][0],robot['u'][1]]

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

    def plot_for_one_robot(self,robot, robot_id):
        x,y,theta,v,w = robot['State']

        # Calculate all fute x and y states
        x_pred, y_pred = self.control_action_to_trajectory(x,y,theta,robot['u'])

        # Save the states that we have been to
        robot['Past_x'].append(x)
        robot['Past_y'].append(y)

        plt.plot(robot['Past_x'],robot['Past_y'],'-o', color=robot['Color'], label="Robot{}".format(robot_id))
        plt.plot(x_pred,y_pred,'-o', alpha=0.2,color=robot['Color'])
        self.plot_robot(x,y,theta)
        #self.plot_safety_cricles(x,y)

    def plot_map(self,robots): 
        plt.subplot(1,2,1)
        plt.cla()
        for robot_id in robots: 
            self.plot_for_one_robot(robots[robot_id], robot_id)
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.grid()
        plt.title("Map")

    def plot_dist(self,robots, N): 
        t_vec = np.linspace(0,N*self.ts,N+1)

        plt.subplot(3,2,2)
        #self.plots["Distance"]
        plt.cla()
        for comb in combinations(range(0,self.nr_of_robots),2):
            x1,y1,theta1,v,w = robots[comb[0]]['State']
            x2,y2,theta2,v,w = robots[comb[1]]['State']   
            dist = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
            self.dist[comb].append(dist)
            plt.plot(t_vec,self.dist[comb], label="Distance for {}".format(comb))
            lim_dist = 1
            plt.plot(t_vec, [lim_dist]*len(self.dist[comb]), label="Limit")

        plt.ylabel("m")
        #plt.xlabel("N")
        plt.legend()
        plt.title("Distance")
        plt.grid()
    
    def plot_vel(self,robots, N): 
        t_vec = np.linspace(0,N*self.ts,N+1)
        t_vec = t_vec.tolist()

        plt.subplot(3,2,4)
        plt.cla()
        for robot_id in robots: 
            robot = robots[robot_id]
            plt.plot(t_vec,robot['Past_v'], '-.',color=robot['Color'], label="Robot{}".format(robot_id))
        plt.ylim(0,2.0)
        #plt.xlabel("N")
        plt.ylabel("m/s")
        plt.title("Velocity")
        plt.legend()
        plt.grid()

        plt.subplot(3,2,6)
        plt.cla()
        for robot_id in robots: 
            robot = robots[robot_id]
            plt.plot(t_vec,robot['Past_w'],'-.', color=robot['Color'], label="Robot{}".format(robot_id))
        plt.ylim(-1.5,1.5)
        plt.xlabel("t")
        plt.ylabel("rad/s")
        plt.title("Angular velocity")
        plt.legend()
        plt.grid()
        
    def get_input(self, robots):
        # Create the input vector
        p = []

        for robot_id in robots: 
            p.extend(robots[robot_id]['State'])
            p.extend(robots[robot_id]['Ref'])

        p.extend(self.weights)
        return p

    def run_one_iteration(self,robots,iteration_step): 
        mpc_input = self.get_input(robots)

        # Call the solver
        t1 = perf_counter_ns()
        solution = self.mng.call(p=mpc_input, initial_guess=[1.0] * (self.nr_of_robots*self.nu*self.N))
        t2 = perf_counter_ns()
        self.time += (t2-t1)/10**6 

        # Get the solver output 
        u_star = solution['solution']
        
        for i in range(0,self.nr_of_robots):
            robots[i]['u'] = u_star[self.nu*self.N*i:self.nu*self.N*(i+1)]

        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])

        self.plot_map(robots)
        self.plot_dist(robots, iteration_step)
        self.plot_vel(robots, iteration_step)
        plt.pause(0.001)
        
        
    def run(self, robots):
        plt.show(block=False)
        plt.tight_layout(pad=3.0)
        plt.pause(5)

        for i in range(0,60+1): 
            self.run_one_iteration(robots,iteration_step=i)
        plt.pause(2)
        print("Avg solvtime: ", self.time/61," ms")
        

if __name__=="__main__": 
    
    """
    # Case 1 - Crossing
    r_model = RobotModelData(nr_of_robots=2, nx=5, q = 100, qtheta=100, r=10, qN=10, qaccW=10, qaccV=20)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    
    avoid.run(robots)
    avoid.mng.kill()
    
    
    # Case 2 - Towards eachother
    r_model = RobotModelData(nr_of_robots=2, nx=5)
    avoid = CollisionAvoidance(r_model)
    nx =5
    traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=2,y=0,theta=-cs.pi,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    avoid.run(robots)
    avoid.mng.kill()
    
    
    # Case 3 - Behind eachother
    r_model = RobotModelData(nr_of_robots=2, nx=5, q = 100, r=10, qN=100, qaccW=10, qaccV=100)
    avoid = CollisionAvoidance(r_model)
    nx = 5
    traj1 = generate_straight_trajectory(x=-1,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=-2.1,y=0,theta=0,v=1.3,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    avoid.run(robots)
    avoid.mng.kill()
    """

    # Case 4 - Multiple Robots
    r_model = RobotModelData(nr_of_robots=5, nx=5, q = 100, qtheta=100, r=10, qN=10, qaccW=10, qaccV=20)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=60) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=3,y=0,theta=-cs.pi,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    traj3 = generate_straight_trajectory(x=1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    traj4 = generate_straight_trajectory(x=-1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=60) # Trajectory from x=0,y=-1 driving straight up
    traj5 = generate_straight_trajectory(x=-3,y=2,theta=0,v=1,ts=0.1,N=60) # Trajectory from x=-1, y=0 driving straight to the right
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:20*nx+nx], 'Remainder': traj3[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:20*nx+nx], 'Remainder': traj4[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:20*nx+nx], 'Remainder': traj5[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y'}
    
    avoid.run(robots)
    avoid.mng.kill()

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

    