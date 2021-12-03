from os import stat
import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from numpy.lib.arraypad import pad
from function_lib import model, generate_straight_trajectory, compute_polytope_halfspaces, predict, padded_square, unpadded_square, polygon_to_eqs
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
from RobotModelData import RobotModelData
from plotter import Plotter
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
        self.ustar  = {i: [] for i in range(r_model.nr_of_robots)}

        # Plotter object
        self.plotter = Plotter(name='distributed',r_model=r_model)

        # Create the solver and open a tcp port to it 
        self.mng = og.tcp.OptimizerTcpManager("distributed1/distributed_solver_{}".format(self.nr_of_robots))
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

    def predicted_states_from_u(self,x,y,theta,u): 
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

        
    def get_input(self, state, ref, obstacles, predicted_states, robot_id):
        # Create the input vector
        p = []
        p.extend(state)
        p.extend(ref)
        p.extend(self.weights)

        # Get predicted states for the other robot
        for i in predicted_states: 
            if robot_id != i:
                p.extend(predicted_states[i])
        
        for ob in obstacles['Padded']: 
            p.extend(polygon_to_eqs(ob))

        return p

    #select the ref control signals for all robots 
    def get_control_signals_from_ref(self, robots):
        #might be slow
        u_p = {i: [] for i in range(len(robots))}
        for robot_id in robots: 
            [u_p[robot_id].extend(robots[robot_id]['Ref'][self.nx*i+3:self.nx*(i+1)]) for i in range(self.N)]
        return u_p


    def shift_left_and_append_last(self,dikt):   
        # Shift reference once step to the left
        for i in range(len(dikt)):
            dikt[i].extend(dikt[i][-self.nu:]) 
            dikt[i] = dikt[i][self.nu:]

    def distributed_algorithm(self,robots, obstacles, predicted_states):
        predicted_states_temp = predicted_states.copy()

        #Update the old control signals to last iterations signals
        u_p_old = self.get_control_signals_from_ref(robots)

        w = .65 # 
        pmax = 1
        epsilon = .01
        if len(self.ustar[0])<1:
            self.ustar = u_p_old.copy()

        self.shift_left_and_append_last(self.ustar)
        

        times = [0]*self.nr_of_robots
        t3 = perf_counter_ns()
        for i in range(pmax):
            K = 0
            for robot_id in robots: 
                if not robots[robot_id]['dyn_obj']:
                    state = robots[robot_id]['State']
                    ref = robots[robot_id]['Ref']
                    mpc_input = self.get_input(state,ref,obstacles, predicted_states, robot_id)

                    # Call the solver
                    t1 = perf_counter_ns()
                    
                    #use ustar - prev best solution as init guess
                    solution = self.mng.call(p=mpc_input, initial_guess=self.ustar[robot_id])
                    t2 = perf_counter_ns()
                    self.time += (t2-t1)/10**6 
                    self.time_vec.append((t2-t1)/10**6 )

                    times[robot_id] += (t2-t1)/10**6

                    # Get the solver output 
                    self.ustar[robot_id] = solution['solution'] 
                    #modify the output to not be to far from the previous
                    u_p = [w*self.ustar[robot_id][j] + (1-w)*u_p_old[robot_id][j] for j in range(self.N*self.nu)]
                    
                    K = max(K, max([abs(u_p[j] - u_p_old[robot_id][j]) for j in range(self.N*self.nu)]))
                    
                    #to be used in next it
                    u_p_old[robot_id] = u_p

                    # Predict future state
                    x,y,theta = state[0], state[1],state[2]

                else:
                    ustar = {0:[]} 
                    [ustar[0].extend(robots[robot_id]['Ref'][self.nx*i+3:self.nx*(i+1)]) for i in range(self.N)]
                    self.ustar[robot_id] = ustar[0]
                    x,y,theta = robots[robot_id]['Ref'][0:3]
                states = self.predicted_states_from_u(x,y,theta,self.ustar[robot_id])
                
                predicted_states_temp[robot_id] = states

                robots[robot_id]['u'] = self.ustar[robot_id]

            predicted_states.update(predicted_states_temp)
            if K < epsilon:
                break
        
        t4 = perf_counter_ns()
        self.time2 += (t4-t3)/10**6 
        self.time_vec2.append((t4-t3)/10**6 )
        self.time_vec3[0].append(times[0])
        self.time_vec3[1].append(times[1])

    def run_one_iteration(self,robots,obstacles,iteration_step,predicted_states): 
        #run the distributed algorithm
        self.distributed_algorithm(robots,  obstacles, predicted_states)
        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])        
        # Call the plotter object to plot everyting
        self.plotter.plot(robots, obstacles, iteration_step)
        
        
    def run(self, robots, obstacles, sim_steps, predicted_states):
        for i in range(0,sim_steps):
            self.run_one_iteration(robots,obstacles,i,predicted_states)
        self.plotter.stop()
        self.plotter.plot_computation_time(self.time_vec)
        

if __name__=="__main__":
    
    
    case_nr = 2
    obstacle_case = 0

    sim_steps = 60
    N_steps = 180
    #r_model = RobotModelData(nx=5, q = 5, qtheta = 10, qobs=400, r=20, qN=200, qaccW=5, qthetaN = 200, qaccV=15, N=20) # w = .75, pmax = 10, epsilon = .01 ref
    #r_model = RobotModelData(nx=5, q =250, qtheta = 10, qobs=400, r=30, qN=150, qaccW=.5, qthetaN = 20, qaccV=15, N=20) # w = .75, pmax = 5, epsilon = .01 line
    r_model = RobotModelData(nx=5, q = 250, qtheta = 10, qobs=2000, r=20, qN=2000, qpol=2000, qaccW=.5, qthetaN = 20, qaccV=15, N=20) # under development for better performance

    obstacles = {}
    obstacles['Unpadded'] =  [None, None, None, None, None]
    obstacles['Padded'] = [None, None, None, None, None]

    obstacles['Boundaries'] =  Polygon([[-4.5, -4.5], [4.5, -4.5], [4.5, 4.5], [-4.5, 4.5]]) 
    obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': False}

    # obs case1, 4 obstacles in center
    if obstacle_case == 1:       
        obstacles['Unpadded'] =  [unpadded_square(-1,-1,1,1), unpadded_square(1,-1,1,1), unpadded_square(1,1,1,1), unpadded_square(-1,1,1,1), None]
        obstacles['Padded'] = [padded_square(-1,-1,1,1, 0.5), padded_square(1,-1,1,1, 0.5), padded_square(1,1,1,1, 0.5), padded_square(-1,1,1,1, 0.5), None]
    
    # obs case2, 1 obstacle in center
    if obstacle_case == 2:
        obstacles['Unpadded'] =  [unpadded_square(0,0,1,1), None, None, None, None]
        obstacles['Padded'] = [padded_square(0,0,1,1, 0.5), None, None, None, None]


    if case_nr == 1:
        #intersection 2 robots
        r_model.nr_of_robots=2
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=0,y=-3,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        
    

    if case_nr == 2:
        #collition, head on,  2 robots
        r_model.nr_of_robots=2
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-3, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=4,y=0,theta=-cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=3,y=-.1 driving straight to the left

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        


    if case_nr == 3:
        # Case 3 - Behind eachother
        r_model.nr_of_robots=2
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-2,y=0,theta=0,v=0.8,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1.3,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        
        nx = 5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        

    

    if case_nr == 4:
        # Case 4 - Multiple Robots mix
        r_model.nr_of_robots=5
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-4, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=4,y=0,theta=-cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=4,y=0 driving straight to the left
        traj3 = generate_straight_trajectory(x=1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=1,y=-2 driving straight up
        traj4 = generate_straight_trajectory(x=-1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1,y=-2 driving straight up
        traj5 = generate_straight_trajectory(x=-4,y=2,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-, y=2 driving straight to the right
   
        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:r_model.N*nx+nx], 'Remainder': traj3[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:r_model.N*nx+nx], 'Remainder': traj4[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:r_model.N*nx+nx], 'Remainder': traj5[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y','dyn_obj':False}
        

    if case_nr == 5:
        #intersection 2 robots + 9 robots
        r_model.nr_of_robots=11
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=0,y=-3,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        extra_traj = [generate_straight_trajectory(x=-3*i,y=-4,theta=0,v=1,ts=0.1,N=N_steps) for i in range(r_model.nr_of_robots-2)] # Trajectory from x=-3, y=-4 driving straight to the right

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        for i in range(len(extra_traj)):
            robots[i+2] = {"State": extra_traj[i][:nx], 'Ref': extra_traj[i][nx:r_model.N*nx+nx], 'Remainder': extra_traj[i][r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        
    
    
    if case_nr == 6:
    # Case 6 - Multiple Robots
        r_model.nr_of_robots=10
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj2 = generate_straight_trajectory(x=0,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) 
        traj3 = generate_straight_trajectory(x=3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj4 = generate_straight_trajectory(x=4,y=0,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj5 = generate_straight_trajectory(x=4,y=-1,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj6 = generate_straight_trajectory(x=1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj7 = generate_straight_trajectory(x=-1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj8 = generate_straight_trajectory(x=-5,y=2,theta=0,v=1,ts=0.1,N=N_steps)
        traj9 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps)
        traj10 = generate_straight_trajectory(x=-4,y=-3,theta=0,v=1,ts=0.1,N=N_steps)

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:r_model.N*nx+nx], 'Remainder': traj3[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:r_model.N*nx+nx], 'Remainder': traj4[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:r_model.N*nx+nx], 'Remainder': traj5[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[5] = {"State": traj6[:nx], 'Ref': traj6[nx:r_model.N*nx+nx], 'Remainder': traj6[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        robots[6] = {"State": traj7[:nx], 'Ref': traj7[nx:r_model.N*nx+nx], 'Remainder': traj7[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        robots[7] = {"State": traj8[:nx], 'Ref': traj8[nx:r_model.N*nx+nx], 'Remainder': traj8[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[8] = {"State": traj9[:nx], 'Ref': traj9[nx:r_model.N*nx+nx], 'Remainder': traj9[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[9] = {"State": traj10[:nx], 'Ref': traj10[nx:r_model.N*nx+nx], 'Remainder': traj10[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
    
    
    if case_nr == 7:
    # Case 6 - Multiple Robots and dyn obj
        r_model.nr_of_robots=11
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=-3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj2 = generate_straight_trajectory(x=0,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) 
        traj3 = generate_straight_trajectory(x=3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj4 = generate_straight_trajectory(x=4,y=0,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj5 = generate_straight_trajectory(x=4,y=-1,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj6 = generate_straight_trajectory(x=1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj7 = generate_straight_trajectory(x=-1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj8 = generate_straight_trajectory(x=-5,y=2,theta=0,v=1,ts=0.1,N=N_steps)
        traj9 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps)
        traj10 = generate_straight_trajectory(x=-4,y=-3,theta=0,v=1,ts=0.1,N=N_steps)
        traj_dyn = generate_straight_trajectory(x=-3,y=-3,theta=cs.pi/4,v=1.4,ts=0.1,N=N_steps)

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:r_model.N*nx+nx], 'Remainder': traj3[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:r_model.N*nx+nx], 'Remainder': traj4[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:r_model.N*nx+nx], 'Remainder': traj5[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[5] = {"State": traj6[:nx], 'Ref': traj6[nx:r_model.N*nx+nx], 'Remainder': traj6[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        robots[6] = {"State": traj7[:nx], 'Ref': traj7[nx:r_model.N*nx+nx], 'Remainder': traj7[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm','dyn_obj':False}
        robots[7] = {"State": traj8[:nx], 'Ref': traj8[nx:r_model.N*nx+nx], 'Remainder': traj8[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[8] = {"State": traj9[:nx], 'Ref': traj9[nx:r_model.N*nx+nx], 'Remainder': traj9[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[9] = {"State": traj10[:nx], 'Ref': traj10[nx:r_model.N*nx+nx], 'Remainder': traj10[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g','dyn_obj':False}
        robots[10] = {"State": traj_dyn[:nx], 'Ref': traj_dyn[nx:r_model.N*nx+nx], 'Remainder': traj_dyn[r_model.N*nx+nx:], 'u': [], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'k','dyn_obj':True}


    if case_nr == 8:
    #intersection 2 robots
        r_model.nr_of_robots=4
        avoid = CollisionAvoidance(r_model)
        traj1 = generate_straight_trajectory(x=0,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=4,y=0,theta=cs.pi,v=1,ts=0.1,N=40) # Trajectory from x=0,y=-1 driving straight upc
        traj2.extend(generate_straight_trajectory(x=0,y=0,theta=cs.pi/2,v=1,ts=0.1,N=N_steps-40)) # Trajectory from x=0,y=-1 driving straight up
        traj3 = generate_straight_trajectory(x=0,y=-3.5,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj4 = generate_straight_trajectory(x=0,y=-5,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right

        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:r_model.N*nx+nx], 'Remainder': traj1[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r','dyn_obj':False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:r_model.N*nx+nx], 'Remainder': traj2[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:r_model.N*nx+nx], 'Remainder': traj3[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:r_model.N*nx+nx], 'Remainder': traj4[r_model.N*nx+nx:], 'u': [0,0], 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b','dyn_obj':False}

        
        
    u = avoid.get_control_signals_from_ref(robots)
    #x = robots[i]['Ref'][0]
    predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}
#    predicted_states = {i: [0]*r_model.N*2 for i in range(r_model.nr_of_robots)}
    avoid.run(robots, obstacles, sim_steps, predicted_states)
    avoid.mng.kill()
    