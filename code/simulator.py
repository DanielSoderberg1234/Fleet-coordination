import casadi.casadi as cs
from numpy.lib.arraypad import pad
from function_lib import model, generate_straight_trajectory, generate_turn_right_trajectory, predict, padded_square, unpadded_square, polygon_to_eqs, dist_to_ref
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
from RobotModelData import RobotModelData
from shapely.geometry import Polygon
from plotter import Plotter
import numpy as np
import random


class Simulator: 
    def __init__(self, r_model: RobotModelData, centralized, distributed):
        # Load parameters 
        self.nr_of_robots = r_model.nr_of_robots
        self.max_nr_of_robots = r_model.max_nr_of_robots
        self.nx = r_model.nx
        self.nu = r_model.nu 
        self.N = r_model.N 
        self.ts = r_model.ts 
        self.weights = r_model.get_weights()
        self.ustar  = {i: [] for i in range(r_model.nr_of_robots)}

        self.centralized = centralized
        self.distributed = distributed
        
        if centralized:
            self.plotter = Plotter(name='centralized',r_model=r_model)
            self.mng = og.tcp.OptimizerTcpManager('collision_avoidance/robot_{}_solver'.format(self.nr_of_robots))
            self.mng.start()
        elif distributed: 
            self.plotter = Plotter(name='distributed',r_model=r_model)
            self.mng = og.tcp.OptimizerTcpManager("distributed1/distributed_solver_{}".format(self.max_nr_of_robots))
            self.mng.start()

        # Save distance for each combination
        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []

        # Time 
        self.time = 0
        self.time_vec = []

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
        #robot['u'][0] += random.random()/10
        #robot['u'][1] += random.random()/10
        x,y,theta = model(x,y,theta,robot['u'][:2],self.ts)
        robot['State'] = [x,y,theta,robot['u'][0],robot['u'][1]]

    def update_ref(self,robot):
        # Shift reference once step to the left
        robot['Ref'][:self.nx*(self.N-1)] = robot['Ref'][self.nx:]

        # If there are more points in trajectory, append them to the end of the reference 
        # else, we are getting closer to the end of the trajectory 
        if len(robot['Remainder']) > 0:
            robot['Ref'][-self.nx:] = robot['Remainder'][:self.nx]
            del robot['Remainder'][:self.nx]

    def update_dynamic_obstacle(self,obstacles): 
        # Take one step for the ellipse with velocity in each coordinate times sampling time
        obstacles['Dynamic']['center'][0] += self.ts*obstacles['Dynamic']['vel'][0]
        obstacles['Dynamic']['center'][1] += self.ts*obstacles['Dynamic']['vel'][1]
        
    def get_centralized_input(self, robots, obstacles):
        # Create the input vector
        p = []

        # Add state and reference for each robot to the input vecotr
        for robot_id in robots: 
            p.extend(robots[robot_id]['State'])
            p.extend(robots[robot_id]['Ref'])

        # Append the weights
        p.extend(self.weights)

        # Append the parameters for each padded polygon to the input vector
        for ob in obstacles['Padded']: 
            p.extend(polygon_to_eqs(ob))

        # Append equation for boundaries
        p.extend(polygon_to_eqs(obstacles['Boundaries']))

        # Append parameters for the ellipse: a,b,phi,predicted centers
        p.append(obstacles['Dynamic']['a']+obstacles['Dynamic']['apad'])
        p.append(obstacles['Dynamic']['b']+obstacles['Dynamic']['bpad'])
        p.append(obstacles['Dynamic']['phi'])
        p.extend(predict(obstacles['Dynamic']['center'][0],
                        obstacles['Dynamic']['center'][1],
                        obstacles['Dynamic']['vel'][0],
                        obstacles['Dynamic']['vel'][1],
                        self.N,
                         self.ts))

        return p

    def get_initial_centralized_guess(self, robots): 
        init_guess = []
        for robot_id in robots: 
            init_guess.extend(robots[robot_id]['u'])
        return init_guess

    def step_centralized(self,robots,obstacles,iteration_step): 
        mpc_input = self.get_centralized_input(robots, obstacles)
        init_guess = self.get_initial_centralized_guess(robots)

        # Call the solver and time it
        t1 = perf_counter_ns()
        solution = self.mng.call(p=mpc_input, initial_guess=init_guess)
        t2 = perf_counter_ns()
        self.time += (t2-t1)/10**6 
        self.time_vec.append((t2-t1)/10**6 )

        # Get the solver output 
        u_star = solution['solution']
        
        # Slice the input to each robot to get their control signals
        for i in range(0,self.nr_of_robots):
            robots[i]['u'] = u_star[self.nu*self.N*i:self.nu*self.N*(i+1)]
        
        # Update states and references for each robot
        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])
        self.update_dynamic_obstacle(obstacles)

        # Call the plotter object to plot everyting
        self.plotter.plot(robots, obstacles, iteration_step)

    def get_distributed_input(self, state, ref, obstacles, predicted_states, robot_id):
        # Create the input vector
        p = []
        p.extend(state)
        p.extend(ref)
        p.extend(self.weights)

        # Get predicted states for the other robot
        for i in predicted_states:
            if i != robot_id:
                p.extend(predicted_states[i])
        # Fill in with  zero trejectories for missing robots
        p.extend([0]*2*self.N*(self.max_nr_of_robots - self.nr_of_robots))

        p.extend([1]*(self.nr_of_robots-1))
        p.extend([0]*(self.max_nr_of_robots - self.nr_of_robots))
        
        for ob in obstacles['Padded']: 
            p.extend(polygon_to_eqs(ob))

        # Append equation for boundaries
        p.extend(polygon_to_eqs(obstacles['Boundaries']))

        # Append parameters for the ellipse: a,b,phi,predicted centers
        p.append(obstacles['Dynamic']['a']+obstacles['Dynamic']['apad'])
        p.append(obstacles['Dynamic']['b']+obstacles['Dynamic']['bpad'])
        p.append(obstacles['Dynamic']['phi'])
        p.extend(predict(obstacles['Dynamic']['center'][0],
                        obstacles['Dynamic']['center'][1],
                        obstacles['Dynamic']['vel'][0],
                        obstacles['Dynamic']['vel'][1],
                        self.N,
                        self.ts))

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
        for i in range(pmax):
            K = 0
            for robot_id in robots: 
                if not robots[robot_id]['dyn_obj']:
                    state = robots[robot_id]['State']
                    ref = robots[robot_id]['Ref']
                    mpc_input = self.get_distributed_input(state,ref,obstacles, predicted_states, robot_id)

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

    def step_distributed(self,robots,obstacles,iteration_step,predicted_states): 
        #run the distributed algorithm
        self.distributed_algorithm(robots,  obstacles, predicted_states)
        for robot_id in robots: 
            self.update_state(robots[robot_id])
            self.update_ref(robots[robot_id])        
        self.update_dynamic_obstacle(obstacles)
        # Call the plotter object to plot everyting
        self.plotter.plot(robots, obstacles, iteration_step)
        
        
    def run(self, robots, obstacles, sim_steps, predicted_states):
        tot_dist = []
        # Run the simulation for a number of steps
        if self.centralized:
            print('-------centralized-------')
            for i in range(0,sim_steps): 
                self.step_centralized(robots,obstacles,iteration_step=i)
                tot_dist.append(dist_to_ref(robots))
        elif self.distributed: 
            print('-------distributed-------')
            for i in range(0,sim_steps): 
                self.step_distributed(robots,obstacles,i,predicted_states)
                tot_dist.append(dist_to_ref(robots))
        total_dist_robot = [sum(x) for x in zip(*tot_dist)]
        total_dist_robot = [total_dist_robot[i]*0.1 for i in range(len(total_dist_robot))]
        print('total dist per robot')
        print(total_dist_robot)
        print('total dist summed')
        print(sum(total_dist_robot))
        print('mean: ',np.mean(total_dist_robot),'var: ',np.var(total_dist_robot))
        self.plotter.stop()
        self.plotter.plot_computation_time(self.time_vec)


if __name__=="__main__": 

    nx =5
    nu = 2
    N = 20
    sim_steps = 70
    centralized = True
    distributed = False
    case_nr = 2

    q_lines = 10

    

    if case_nr == 1:
        N_steps = 60 
        r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=0, q=q_lines)
        avoid = Simulator(r_model, centralized=centralized, distributed=distributed)
        traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=0,y=-2.99,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up

        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        
        obstacles = {}
        obstacles['Unpadded'] =  [unpadded_square(-1,-1,1,1), unpadded_square(1,-1,1,1), unpadded_square(1,1,1,1), unpadded_square(-1,1,1,1), None]
        obstacles['Padded'] = [padded_square(-1,-1,1,1, 0.5), padded_square(1,-1,1,1, 0.5), padded_square(1,1,1,1, 0.5), padded_square(-1,1,1,1, 0.5), None]
        obstacles['Boundaries'] =  Polygon([[-4, -4], [4, -4], [4, 4], [-4, 4]])   
        obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': False}

        u = avoid.get_control_signals_from_ref(robots)
        predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}

        avoid.run(robots, obstacles, sim_steps, predicted_states)
        avoid.mng.kill()


    if case_nr == 2: 
        N_steps = 70
        r_model = RobotModelData(nr_of_robots=10, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=200, q=q_lines)
        avoid = Simulator(r_model, centralized=centralized, distributed=distributed)
        traj1 = generate_straight_trajectory(x=-2.5,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj2 = generate_straight_trajectory(x=0,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps) 
        traj3 = generate_straight_trajectory(x=3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj4 = generate_straight_trajectory(x=4,y=1,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj5 = generate_straight_trajectory(x=4,y=-1,theta=cs.pi,v=1,ts=0.1,N=N_steps)
        traj6 = generate_straight_trajectory(x=1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj7 = generate_straight_trajectory(x=-1,y=-4,theta=cs.pi/2,v=1,ts=0.1,N=N_steps)
        traj8 = generate_straight_trajectory(x=-4,y=3,theta=0,v=1,ts=0.1,N=N_steps)
        traj9 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps)
        traj10 = generate_straight_trajectory(x=-4,y=-3,theta=0,v=1,ts=0.1,N=N_steps)

        obstacles = {}
        obstacles['Unpadded'] =  [None, None, None, None, None]
        obstacles['Padded'] = [None, None, None, None, None]
        obstacles['Boundaries'] =  Polygon([[-4.5, -4.5], [4.5, -4.5], [4.5, 4.5], [-4.5, 4.5]]) 
        obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': True}
        
        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:N*nx+nx], 'Remainder': traj3[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:N*nx+nx], 'Remainder': traj4[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:N*nx+nx], 'Remainder': traj5[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        robots[5] = {"State": traj6[:nx], 'Ref': traj6[nx:N*nx+nx], 'Remainder': traj6[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm', 'dyn_obj': False}
        robots[6] = {"State": traj7[:nx], 'Ref': traj7[nx:N*nx+nx], 'Remainder': traj7[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm', 'dyn_obj': False}
        robots[7] = {"State": traj8[:nx], 'Ref': traj8[nx:N*nx+nx], 'Remainder': traj8[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g', 'dyn_obj': False}
        robots[8] = {"State": traj9[:nx], 'Ref': traj9[nx:N*nx+nx], 'Remainder': traj9[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g', 'dyn_obj': False}
        robots[9] = {"State": traj10[:nx], 'Ref': traj10[nx:N*nx+nx], 'Remainder': traj10[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g', 'dyn_obj': False}
        
        u = avoid.get_control_signals_from_ref(robots)
        predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}

        avoid.run(robots, obstacles, sim_steps, predicted_states)
        avoid.mng.kill()


    if case_nr == 3:
        N_steps = 90 
        r_model = RobotModelData(nr_of_robots=5, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=0, q=q_lines)
        avoid = Simulator(r_model, centralized=centralized, distributed=distributed)
        traj1 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=-2.5,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj3 = generate_straight_trajectory(x=-1.0,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj4 = generate_straight_trajectory(x=0.5,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj5 = generate_turn_right_trajectory(x=0,y=-3.5,theta=cs.pi/2,v=1,ts=0.1,N1=35, N2=N_steps) # Trajectory from x=0,y=-1 driving straight up
        
        nx =5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:20*nx+nx], 'Remainder': traj3[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g', 'dyn_obj': False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:20*nx+nx], 'Remainder': traj4[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm', 'dyn_obj': False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:20*nx+nx], 'Remainder': traj5[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y', 'dyn_obj': False}
        
        obstacles = {}
        obstacles['Unpadded'] =  [None,None, None, None, None]
        obstacles['Padded'] = [None, None, None, None, None]
        obstacles['Boundaries'] =  Polygon([[-9, -9], [9, -9], [9, 9], [-9, 9]]) 
        obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': False}

        u = avoid.get_control_signals_from_ref(robots)
        predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}

        avoid.run(robots, obstacles, sim_steps, predicted_states)
        avoid.mng.kill()

    if case_nr == 4: 
        # Case 2 - 5 Robots
        N_steps = 70
        r_model = RobotModelData(nr_of_robots=5, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=200, q=q_lines)
        avoid = Simulator(r_model, centralized=centralized, distributed=distributed)
        traj1 = generate_straight_trajectory(x=-4,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=4,y=0,theta=-cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj3 = generate_straight_trajectory(x=1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj4 = generate_straight_trajectory(x=-1,y=-2,theta=cs.pi/2,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up
        traj5 = generate_straight_trajectory(x=-4,y=2,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right

        obstacles = {}
        obstacles['Unpadded'] =  [None, None, None, None, None]
        obstacles['Padded'] = [None, None, None, None, None]
        obstacles['Boundaries'] =  Polygon([[-4.5, -4.5], [4.5, -4.5], [4.5, 4.5], [-4.5, 4.5]]) 
        obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': True}
        
        nx = 5
        nx = 5
        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:20*nx+nx], 'Remainder': traj1[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:20*nx+nx], 'Remainder': traj2[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:20*nx+nx], 'Remainder': traj3[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g', 'dyn_obj': False}
        robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:20*nx+nx], 'Remainder': traj4[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm', 'dyn_obj': False}
        robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:20*nx+nx], 'Remainder': traj5[20*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y', 'dyn_obj': False}
        
        u = avoid.get_control_signals_from_ref(robots)
        predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}

        avoid.run(robots, obstacles, sim_steps, predicted_states)
        avoid.mng.kill()

    if case_nr == 5:
        N_steps = 60 
        r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=0, q=q_lines)
        avoid = Simulator(r_model, centralized=centralized, distributed=distributed)
        traj1 = generate_straight_trajectory(x=-3,y=0,theta=0,v=1,ts=0.1,N=N_steps) # Trajectory from x=-1, y=0 driving straight to the right
        traj2 = generate_straight_trajectory(x=3,y=0,theta=cs.pi,v=1,ts=0.1,N=N_steps) # Trajectory from x=0,y=-1 driving straight up

        robots = {}
        robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r', 'dyn_obj': False}
        robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b', 'dyn_obj': False}
        
        obstacles = {}
        obstacles['Unpadded'] =  [unpadded_square(0,2,8,2.5), unpadded_square(0,-2,8,2.5), None, None, None]
        obstacles['Padded'] = [padded_square(0,2,8,2.5, 0.5), padded_square(0,-2,8,2.5, 0.5), None, None, None]
        obstacles['Boundaries'] =  Polygon([[-8, -8], [8, -8], [8, 8], [-8, 8]])   
        obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': False}

        u = avoid.get_control_signals_from_ref(robots)
        predicted_states = {i: avoid.predicted_states_from_u(robots[i]['Ref'][0],robots[i]['Ref'][1],robots[i]['Ref'][2],u[i]) for i in range(r_model.nr_of_robots)}

        avoid.run(robots, obstacles, sim_steps, predicted_states)
        avoid.mng.kill()
            
            
    