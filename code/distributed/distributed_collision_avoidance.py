from os import stat
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
        self.mng = og.tcp.OptimizerTcpManager("distributed1/distributed_solver_{}_robots".format(self.nr_of_robots))
        self.mng.start()
        self.mng.ping()

        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []

        # Time 
        self.time = 0
        self.time2 = 0
        self.time_vec = []
        self.time_vec2 = []
        self.time_vec3 = {0: [], 1: []}


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

    def predicted_states(self,x,y,theta,u): 
        # Get the linear and angular velocities
        v = u[0::2]
        w = u[1::2]

        # Create a list of x and y states
        states = []

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],self.ts)
            states.extend([x,y])

        return states

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

        # Get the reference 
        ref = robot['Ref']

        x_ref = [x]
        y_ref = [y]

        x_ref.extend(ref[0::5])
        y_ref.extend(ref[1::5])

        plt.plot(robot['Past_x'],robot['Past_y'],'-o', color=robot['Color'], label="Robot{}".format(robot_id))
        plt.plot(x_pred,y_pred,'-o', alpha=0.2,color=robot['Color'])
        plt.plot(x_ref,y_ref,'-x',color='k',alpha=1)
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
        
    def get_input(self, state, ref, predicted_states, robot_id):
        # Create the input vector
        p = []
        p.extend(state)
        p.extend(ref)
        p.extend(self.weights)

        # Get predicted states for the other robot
        for i in predicted_states: 
            if robot_id != i:
                p.extend(predicted_states[i])

        return p

    def distributed_algorithm(self,robots,predicted_states):
        
        u_p_old = [robots[robot_id]['Ref'][3:5] for robot_id in robots]

        w = 0.9
        pmax = 20
        epsilon = 0.01

        times = [0]*self.nr_of_robots

        t3 = perf_counter_ns()
        for i in range(0,pmax):
            K = 0
            for robot_id in robots: 
                state = robots[robot_id]['State']
                ref = robots[robot_id]['Ref']
                mpc_input = self.get_input(state,ref, predicted_states, robot_id)

                # Call the solver
                t1 = perf_counter_ns()
                solution = self.mng.call(p=mpc_input, initial_guess=[1.0] * (self.nu*self.N))
                t2 = perf_counter_ns()
                self.time += (t2-t1)/10**6 
                self.time_vec.append((t2-t1)/10**6 )

                times[robot_id] += (t2-t1)/10**6

                # Get the solver output 
                ustar = solution['solution'] 

                u_p = [w*ustar[j] + (1-w)*u_p_old[robot_id][j] for j in range(self.nu)]
                
                K = max(K, max([u_p[j] - u_p_old[robot_id][j] for j in range(self.nu)]))
                
                u_p_old[robot_id] = u_p

                # Predict future state
                x,y,theta = state[0], state[1],state[2]

                states = self.predicted_states(x,y,theta,ustar)
            
                predicted_states[robot_id] = states

                robots[robot_id]['u'] = u_p

            if K < epsilon:
                break
        t4 = perf_counter_ns()
        self.time2 += (t4-t3)/10**6 
        self.time_vec2.append((t4-t3)/10**6 )
        self.time_vec3[0].append(times[0])
        self.time_vec3[1].append(times[1])

    def run_one_iteration(self,robots,predicted_states,iteration_step): 
        self.distributed_algorithm(robots, predicted_states)

        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])

        self.plot_map(robots)
        self.plot_dist(robots, iteration_step)
        self.plot_vel(robots, iteration_step)
        plt.pause(0.001)
        
        
    def run(self, robots, predicted_states):
        plt.show(block=False)
        plt.tight_layout(pad=3.0)
        

        for i in range(0,40+1): 
            self.run_one_iteration(robots,predicted_states,iteration_step=i)
        plt.pause(2)
       
        plt.close()

        plt.subplot(2,2,1)
        plt.plot(self.time_vec,'-o')
        plt.ylim(0,50)
        plt.title("Calculation Time")
        plt.xlabel("N")
        plt.ylabel('ms')

        plt.subplot(2,2,2)
        plt.plot(self.time_vec2,'-o')
        plt.ylim(0,400)
        plt.title("Calculation Time")
        plt.xlabel("N")
        plt.ylabel('ms')

        plt.subplot(2,2,3)
        plt.plot(self.time_vec3[0],'-o')
        plt.ylim(0,400)
        plt.title("Calculation Time Robot 1")
        plt.xlabel("N")
        plt.ylabel('ms')

        plt.subplot(2,2,4)
        plt.plot(self.time_vec3[1],'-o')
        plt.ylim(0,400)
        plt.title("Calculation Time Robot 2")
        plt.xlabel("N")
        plt.ylabel('ms')

        plt.show()
        

if __name__=="__main__": 
    
    
    case_nr = 1

    if case_nr == 1:
        r_model = RobotModelData(nr_of_robots=2, nx=5, q = 100, qtheta = 10, qobs=10, r=20, qN=2000, qaccW=5, qaccV=5)
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=1,ts=0.1,N=40) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight up

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
        predicted_states = {0: [0]*20*2, 1: [0]*20*2}
        avoid.run(robots, predicted_states)
        avoid.mng.kill()
    

    if case_nr == 4:
        # Case 4 - Multiple Robots
        N_steps = 60
        r_model = RobotModelData(nr_of_robots=5, nx=5, q=200, qobs=200, r=50, qN=200, qaccW=50, qaccV=50)
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
        
        predicted_states = {i: [0]*20*2 for i in range(5)}
        avoid.run(robots, predicted_states)
        avoid.mng.kill()

    
    
    

    