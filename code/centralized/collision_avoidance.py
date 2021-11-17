import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from numpy.lib.arraypad import pad
from function_lib import model, generate_straight_trajectory, compute_polytope_halfspaces
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
from RobotModelData import RobotModelData
from shapely.geometry import Polygon

"""
 A file for testing the MPC. 
 Calls the MPC and modifies the trajectories multiple times.
 It runs the robots with the assumption that they 
 follow the trajectory exactly. A new MPC call is done at every times instance
 It plots the predicted path, where it's at and a 1.0 m distance constrain.
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
        self.time_vec = []


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
        """
        x,y = robot['State'][:2]
        
        while True: 
            xc,yc = robot['Ref'][:2]

            if cs.sqrt( (xc-x)**2 + (yc-y)**2 ) > 0.2 or ( robot['Ref'][0]==robot['Ref'][5] and robot['Ref'][1]==robot['Ref'][6]):
                break  
        """
        
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

        # Get the reference 
        ref = robot['Ref']

        # Reference
        x_ref = [x]
        y_ref = [y]

        x_ref.extend(ref[0::5])
        y_ref.extend(ref[1::5])

        plt.plot(robot['Past_x'],robot['Past_y'],'-o', color=robot['Color'], label="Robot{}".format(robot_id), alpha=0.8)
        plt.plot(x_pred,y_pred,'-o', alpha=0.2,color=robot['Color'])
        plt.plot(x_ref,y_ref,'-x',color='k',alpha=1)
        self.plot_robot(x,y,theta)
        #self.plot_safety_cricles(x,y)

    def plot_polygon(self,polygon): 
        x,y = polygon.exterior.xy 
        plt.plot(x,y,color='k')

    def plot_map(self,robots, obstacles): 
        plt.subplot(1,2,1)
        plt.cla()
        for robot_id in robots: 
            self.plot_for_one_robot(robots[robot_id], robot_id)

        # Plot objects 
        self.plot_polygon(obstacles['Static'][0])
        self.plot_polygon(obstacles['Boundaries'][0])
        
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

    def polygon_to_eqs(self, polygon): 
        vertices = polygon.exterior.coords[:-1]
        A, b = compute_polytope_halfspaces(vertices)
        return [A[0,0],A[0,1],b[0],A[1,0],A[1,1],b[1],A[2,0],A[2,1],b[2],A[3,0],A[3,1],b[3] ]
        
    def get_input(self, robots, obstacles):
        # Create the input vector
        p = []

        for robot_id in robots: 
            p.extend(robots[robot_id]['State'])
            p.extend(robots[robot_id]['Ref'])

        # Append the weights
        p.extend(self.weights)

        eqs = self.polygon_to_eqs(obstacles['Static'][0])
        p.extend(eqs)

        eqs = self.polygon_to_eqs(obstacles['Boundaries'][0])
        p.extend(eqs)

        return p

    def run_one_iteration(self,robots,obstacles,iteration_step): 
        mpc_input = self.get_input(robots, obstacles)

        # Call the solver
        t1 = perf_counter_ns()
        solution = self.mng.call(p=mpc_input, initial_guess=[1.0] * (self.nr_of_robots*self.nu*self.N))
        t2 = perf_counter_ns()
        self.time += (t2-t1)/10**6 
        self.time_vec.append((t2-t1)/10**6 )

        # Get the solver output 
        u_star = solution['solution']
        
        for i in range(0,self.nr_of_robots):
            robots[i]['u'] = u_star[self.nu*self.N*i:self.nu*self.N*(i+1)]

        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])

        self.plot_map(robots, obstacles)
        self.plot_dist(robots, iteration_step)
        self.plot_vel(robots, iteration_step)
        plt.pause(0.001)
        
        
    def run(self, robots, obstacles):
        plt.show(block=False)
        plt.tight_layout(pad=3.0)
        

        for i in range(0,60+1): 
            self.run_one_iteration(robots,obstacles,iteration_step=i)
        plt.pause(2)
        print("Avg solvtime: ", self.time/41," ms")
        plt.close()

        plt.plot(self.time_vec,'-o')
        plt.ylim(0,100)
        plt.title("Calculation Time")
        plt.xlabel("N")
        plt.ylabel('ms')
        plt.show()
        

if __name__=="__main__": 

    obstacles = {}
    obstacles['Static'] =  [Polygon([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]]) ]
    obstacles['Boundaries'] =  [Polygon([[-4, -4], [4, -4], [4, 4], [-4, 4]]) ]
    
    
    # Case 1 - Crossing
    r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=10, qaccV=50)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    
    avoid.run(robots, obstacles)
    avoid.mng.kill()
    
    """
    # Case 2 - Towards eachother
    r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=10, qaccV=50)
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
    r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=10, qaccV=50)
    avoid = CollisionAvoidance(r_model)
    nx = 5
    traj1 = generate_straight_trajectory(x=-1,y=0,theta=0,v=1,ts=0.1,N=60) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=-2.1,y=0,theta=0,v=1.3,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    avoid.run(robots)
    avoid.mng.kill()
    
    
    # Case 4 - Multiple Robots
    N_steps = 60
    r_model = RobotModelData(nr_of_robots=5, nx=5, q=200,qobs=200, r=50, qN=200, qaccW=10, qaccV=50)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=4,y=1,theta=-cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj3 = generate_straight_trajectory(x=1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj4 = generate_straight_trajectory(x=-1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj5 = generate_straight_trajectory(x=-4,y=2,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:20*nx+nx], 'Remainder': traj3[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:20*nx+nx], 'Remainder': traj4[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:20*nx+nx], 'Remainder': traj5[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y'}
    
    avoid.run(robots)
    avoid.mng.kill()
    
    
    # Case 5 - Multiple Robots
    N_steps = 100
    r_model = RobotModelData(nr_of_robots=10, nx=5, q = 10, qtheta=1, r=10, qN=10, qaccW=10, qaccV=20)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-3,y=5,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=0,y=5,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj3 = generate_straight_trajectory(x=3,y=5,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj4 = generate_straight_trajectory(x=5,y=1,theta=cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
    traj5 = generate_straight_trajectory(x=5,y=-1,theta=cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj6 = generate_straight_trajectory(x=-1,y=-5,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj7 = generate_straight_trajectory(x=1,y=-5,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj8 = generate_straight_trajectory(x=-5,y=-3,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj9 = generate_straight_trajectory(x=-5,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    traj10 = generate_straight_trajectory(x=-5,y=3,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
    
    nx = 5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:20*nx+nx], 'Remainder': traj3[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:20*nx+nx], 'Remainder': traj4[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:20*nx+nx], 'Remainder': traj5[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[5] = {"State": traj6[:nx], 'Ref': traj6[nx:20*nx+nx], 'Remainder': traj6[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[6] = {"State": traj7[:nx], 'Ref': traj7[nx:20*nx+nx], 'Remainder': traj7[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[7] = {"State": traj8[:nx], 'Ref': traj8[nx:20*nx+nx], 'Remainder': traj8[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[8] = {"State": traj9[:nx], 'Ref': traj9[nx:20*nx+nx], 'Remainder': traj9[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[9] = {"State": traj10[:nx], 'Ref': traj10[nx:20*nx+nx], 'Remainder': traj10[20*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    
    avoid.run(robots)
    avoid.mng.kill()

    
    
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

    