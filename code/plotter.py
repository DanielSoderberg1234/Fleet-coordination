import matplotlib.pyplot as plt 
import numpy as np 
from function_lib import predict, control_action_to_trajectory
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from RobotModelData import RobotModelData


class Plotter: 
    def __init__(self, name,r_model: RobotModelData):
        # Load parameters 
        self.nr_of_robots = r_model.nr_of_robots
        self.nx = r_model.nx
        self.nu = r_model.nu 
        self.N = r_model.N 
        self.ts = r_model.ts 
        self.weights = r_model.get_weights()
        self.name = name

        self.dist = {}
        for comb in combinations(range(0,self.nr_of_robots),2): 
            self.dist[comb] = []

        plt.show(block=False)
        plt.tight_layout(pad=3.0)
        plt.pause(5)
        

    def stop(self): 
        input("Press enter to close plot")
        plt.close()

    def plot(self, robots, obstacles, iteration_step): 
        self.plot_map(robots, obstacles)
        self.plot_dist(robots, iteration_step)
        self.plot_vel(robots, iteration_step)
        plt.pause(0.001)

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

    def plot_for_one_robot(self,robot, robot_id):
        x,y,theta,v,w = robot['State']

        # Calculate all fute x and y states
        x_pred, y_pred = control_action_to_trajectory(x,y,theta,robot['u'],self.ts)

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

    def plot_polygon(self,polygon): 
        if polygon == None: 
            return
        x,y = polygon.exterior.xy 
        plt.plot(x,y,color='k')

    def plot_ellipses(self,obstacles):
        x,y = obstacles['Dynamic']['center'] 
        centers = [x,y]
        centers += predict(obstacles['Dynamic']['center'][0],obstacles['Dynamic']['center'][1],obstacles['Dynamic']['vel'][0],obstacles['Dynamic']['vel'][1],self.N, self.ts)

        ang = np.linspace(0,2*np.pi, 100)
        for i in range(0, len(centers), 2): 
                c = centers[i:i+2]
                # Equation for an ellipse with an angle: https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
                x = c[0] + obstacles['Dynamic']['a']*np.cos(ang)*np.cos(obstacles['Dynamic']['phi']) - obstacles['Dynamic']['b']*np.sin(ang)*np.sin(obstacles['Dynamic']['phi'])
                y = c[1] + obstacles['Dynamic']['a']*np.cos(ang)*np.sin(obstacles['Dynamic']['phi']) + obstacles['Dynamic']['b']*np.sin(ang)*np.cos(obstacles['Dynamic']['phi'])

                if i == 0: 
                        plt.plot(x,y,color='b')
                else: 
                        plt.plot(x,y,color='b', alpha= 0.2)


    def plot_map(self,robots, obstacles): 
        plt.subplot(1,2,1)
        plt.cla()
        for robot_id in robots: 
            self.plot_for_one_robot(robots[robot_id], robot_id)

        # Plot objects 
        for ob in obstacles['Unpadded']: 
            self.plot_polygon(ob)
        self.plot_polygon(obstacles['Boundaries'])
        
        if obstacles['Dynamic']['active']:
            self.plot_ellipses(obstacles)
            
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
        # loop through all combinations of robots and calculate their distance
        for comb in combinations(range(0,self.nr_of_robots),2):
            x1,y1,theta1,v,w = robots[comb[0]]['State']
            x2,y2,theta2,v,w = robots[comb[1]]['State']   
            dist = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
            self.dist[comb].append(dist)
            plt.plot(t_vec,self.dist[comb], label="Distance for {}".format(comb))
            lim_dist = 1
            plt.plot(t_vec, [lim_dist]*len(self.dist[comb]), label="Limit")

        plt.ylabel("m")
       #plt.legend()
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

    def plot_computation_time(self, time): 
        avg = 0
        for t in time: 
            avg += t 
        avg /= len(time)
        avg = round(avg,3)

        plt.plot(time,'-o', label="Time")
        plt.plot([avg]*len(time), label="Avg time: {} ms".format(avg))
        plt.ylim(0,100)
        plt.title("Calculation Time")
        plt.xlabel("N")
        plt.ylabel('ms')
        plt.legend()
        plt.show()
        plt.close()