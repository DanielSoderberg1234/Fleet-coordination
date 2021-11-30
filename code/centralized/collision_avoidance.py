import casadi.casadi as cs
from numpy.lib.arraypad import pad
from function_lib import model, generate_straight_trajectory, compute_polytope_halfspaces, predict, padded_square, unpadded_square, polygon_to_eqs
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
from RobotModelData import RobotModelData
from shapely.geometry import Polygon
from plotter import Plotter


class CollisionAvoidance: 
    def __init__(self, r_model: RobotModelData):
        # Load parameters 
        self.nr_of_robots = r_model.nr_of_robots
        self.nx = r_model.nx
        self.nu = r_model.nu 
        self.N = r_model.N 
        self.ts = r_model.ts 
        self.weights = r_model.get_weights()

        # Plotter object
        self.plotter = Plotter(name='centralized',r_model=r_model)


        # Create the solver and open a tcp port to it 
        self.mng = og.tcp.OptimizerTcpManager('collision_avoidance/robot_{}_solver'.format(self.nr_of_robots))
        self.mng.start()
        self.mng.ping()

        # Save distance for each combination
        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []

        # Time 
        self.time = 0
        self.time_vec = []

    def update_state(self, robot): 
        x,y,theta,v,w = robot['State']
        robot['Past_v'].append(v)
        robot['Past_w'].append(w)
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
        
    def get_input(self, robots, obstacles):
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

    def get_initial_guess(self, robots): 
        init_guess = []
        for robot_id in robots: 
            init_guess.extend(robots[robot_id]['u'])
        return init_guess

    def run_one_iteration(self,robots,obstacles,iteration_step): 
        mpc_input = self.get_input(robots, obstacles)
        init_guess = self.get_initial_guess(robots)

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
        
        
    def run(self, robots, obstacles, sim_steps):
        # Run the simulation for a number of steps
        for i in range(0,sim_steps): 
            self.run_one_iteration(robots,obstacles,iteration_step=i)
        self.plotter.stop()
        self.plotter.plot_computation_time(self.time_vec)
    
    
if __name__=="__main__": 
    nx =5
    nu = 2
    N = 20
    sim_steps = 70

    """
    # Case 1 - Crossing
    r_model = RobotModelData(nr_of_robots=2, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, qpol=200, qbounds=200, qdyn=0)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-3.5,y=0,theta=0,v=1,ts=0.1,N=70) # Trajectory from x=-1, y=0 driving straight to the right
    traj2 = generate_straight_trajectory(x=0,y=-3.5,theta=cs.pi/2,v=1,ts=0.1,N=70) # Trajectory from x=0,y=-1 driving straight up

    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,0]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    
    obstacles = {}
    obstacles['Unpadded'] =  [unpadded_square(-1,-1,1,1), unpadded_square(1,-1,1,1), unpadded_square(1,1,1,1), unpadded_square(-1,1,1,1), None]
    obstacles['Padded'] = [padded_square(-1,-1,1,1, 0.5), padded_square(1,-1,1,1, 0.5), padded_square(1,1,1,1, 0.5), padded_square(-1,1,1,1, 0.5), None]
    obstacles['Boundaries'] =  Polygon([[-4, -4], [4, -4], [4, 4], [-4, 4]])   
    obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4, 'active': False}

    avoid.run(robots, obstacles, sim_steps)
    avoid.mng.kill()
    """
    
    # Case 2 - 5 Robots
    N_steps = 60
    r_model = RobotModelData(nr_of_robots=5, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50)
    avoid = CollisionAvoidance(r_model)
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
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:N*nx+nx], 'Remainder': traj3[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:N*nx+nx], 'Remainder': traj4[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:N*nx+nx], 'Remainder': traj5[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'y'}
    
    avoid.run(robots, obstacles, sim_steps)
    avoid.mng.kill()
    
    """
    # Case 4 - 10 Robots
    N_steps = 70
    r_model = RobotModelData(nr_of_robots=10, nx=5, qobs=200, r=50, qN=200, qaccW=50, qaccV=50, ts=0.1)
    avoid = CollisionAvoidance(r_model)
    traj1 = generate_straight_trajectory(x=-3,y=4,theta=3*cs.pi/2,v=1,ts=0.1,N=N_steps)
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
    obstacles['Dynamic'] = {'center': [-3,-3], 'a': 0.5, 'b': 0.25, 'vel': [1,1], 'apad': 0.5, 'bpad': 0.5, 'phi': cs.pi/4}
    
    nx =5
    robots = {}
    robots[0] = {"State": traj1[:nx], 'Ref': traj1[nx:N*nx+nx], 'Remainder': traj1[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[1] = {"State": traj2[:nx], 'Ref': traj2[nx:N*nx+nx], 'Remainder': traj2[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[2] = {"State": traj3[:nx], 'Ref': traj3[nx:N*nx+nx], 'Remainder': traj3[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'r'}
    robots[3] = {"State": traj4[:nx], 'Ref': traj4[nx:N*nx+nx], 'Remainder': traj4[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[4] = {"State": traj5[:nx], 'Ref': traj5[nx:N*nx+nx], 'Remainder': traj5[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'b'}
    robots[5] = {"State": traj6[:nx], 'Ref': traj6[nx:N*nx+nx], 'Remainder': traj6[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[6] = {"State": traj7[:nx], 'Ref': traj7[nx:N*nx+nx], 'Remainder': traj7[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'm'}
    robots[7] = {"State": traj8[:nx], 'Ref': traj8[nx:N*nx+nx], 'Remainder': traj8[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[8] = {"State": traj9[:nx], 'Ref': traj9[nx:N*nx+nx], 'Remainder': traj9[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    robots[9] = {"State": traj10[:nx], 'Ref': traj10[nx:N*nx+nx], 'Remainder': traj10[N*nx+nx:], 'u': [1,1]*N, 'Past_x': [], 'Past_y': [], 'Past_v': [], 'Past_w': [], 'Color': 'g'}
    
    avoid.run(robots, obstacles, sim_steps)
    avoid.mng.kill()
    """

    