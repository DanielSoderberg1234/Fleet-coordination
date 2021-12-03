import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model
from itertools import combinations


class MPCGenerator: 
    def __init__(self, nr_of_robots): 
        self.name = "Fleet-collison"
        self.nr_of_robots = nr_of_robots


    def cost_state_ref(self,x,y,theta,xref,yref,thetaref,q,qtheta): 
        # Cost for deviating from the current reference
        return q*( (xref-x)**2 + (yref-y)**2 ) + qtheta*(thetaref-theta)**2


    def cost_turn_left(self,theta,thetaref,qtheta): 
        # Cost for deviating from the current reference only 
        return qtheta*100*cs.fmax(0, -(thetaref-theta))**2


    def cost_lines(self,ref,x,y,N):
        """
        Calculations according to: 
            https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment
        """
        # Extract the references for readability
        x_ref = ref[0::5]
        y_ref = ref[1::5]

        # Define the point we are at
        p = cs.vertcat(x,y)
        
        # Get fist point for the line segment
        s1 = cs.vertcat(x_ref[0],y_ref[0])
        
        # Variable for holding the distance to each line
        dist_vec = None

        # Loop over all possible line segments
        for i in range(1,N):
            # Get the next point
            s2 = cs.vertcat(x_ref[i],y_ref[i])

            #print("\nCurrent linesegment:  {}  <=>  {}  ".format(s1,s2))

            # Calculate t_hat and t_star
            t_hat = cs.dot(p-s1,s2-s1)/((s2[1]-s1[1])**2 + (s2[0]-s1[0])**2 + 1e-16)
            
            t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0)
            
            # Get the closest point from the line s
            st = s1 + t_star*(s2-s1)
            #print("Closes point to:  {}  <=>  {}".format(p,st))
            # Vector from point to the closest point on the line
            dvec = st-p
            
            # Calculate distance
            dist = dvec[0]**2+dvec[1]**2
            #print("The distance is:  {} ".format(dist))
            # Add to distance vector 
            if dist_vec == None: 
                dist_vec = dist
            else: 
                dist_vec = cs.horzcat(dist_vec, dist)

            # Update s1
            s1 = s2
        

        return cs.mmin(dist_vec[:])

    def cost_acceleration(self,u0,u1,qaccV,qaccW): 
        return qaccV*(u1[0]-u0[0])**2 + qaccW*(u1[1]-u0[1])**2
        

    def cost_all_acceleration(self,u,qaccV,qaccW,N): 
        nu = 2
        cost = 0

        u0 = u[:-nu]
        u1 = u[nu:]

        for i in range(0,N-2,2):
            u0i = u0[i:i+2]
            u1i = u1[i:i+2]
            cost += self.cost_acceleration(u0i,u1i,qaccV,qaccW)
            
        return cost

    def cost_inital_acceleration(self,u,u_ref,qaccV,qaccW): 
        return self.cost_acceleration(u,u_ref,qaccV,qaccW)

    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        # But hard constraints on the velocities of the robot
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)
    
    def cost_robot2robot_dist(self,x1,y1,x2,y2,qobs): 
        # Cost for being closer than r to the other robot
        return qobs*cs.fmax(0.0, 1**2 - (x1-x2)**2 - (y1-y2)**2)**2

    def cost_inside_polygon(self,x,y,o,qobs): 
        cost = 0.0
        for i in range(0,5): 
            # Parameter for each object
            ob = o[i*12:(i+1)*12]

            inside = 1
            for j in range(0,12,3):
                h = ob[j:j+3]
                inside *= cs.fmax(0.0, h[2] - h[1]*y - h[0]*x )**2

            cost += qobs*inside

        return cost
    
    def cost_robot_collision(self, x, y, c, k, N, qobs):
        cost = 0.0
        for other_robot_nr in range(self.nr_of_robots-1):
            #2 = nr_of_coord (x,y),                 
            ck = c[2*N*other_robot_nr + k : 2*N*other_robot_nr + k + 2]
            xc,yc = ck[0],ck[1]
            cost += self.cost_robot2robot_dist(x,y,xc,yc,qobs)
        return cost


    def cost_outside_boundaries(self,robots,b,qb): 
        cost = 0.0
        for robot_id in robots: 
            x,y,theta = robots[robot_id]['State']

            outside = 0
            for j in range(0,12,3):
                h = b[j:j+3]
                outside += cs.fmin(0.0, h[2] - h[1]*y - h[0]*x )**2

            cost += qb*outside

        return cost

    def cost_dynamic_obstacle(self, robots, e,j,qpol): 
        cost = 0.0 
        for robot_id in robots: 
            x,y,theta = robots[robot_id]['State']
            a,b, phi = e[0],e[1],e[2]
            centers = e[3:]

            # Equation for ellipse from: https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
            c = centers[j:j+2]
            cost += qpol*cs.fmax(0.0, 1.0 - ((x-c[0])*cs.cos(phi)+(y-c[1])*cs.sin(phi))**2/a**2 - ((x-c[0])*cs.sin(phi)-(y-c[1])*cs.cos(phi))**2/b**2)

        return cost

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 5, 20, 0.1)

        # the 8 last are the weights
        Q = cs.SX.sym('Q',9)

        # Input vector 1 trajectory, N long with nx states in each i=0,1,2,..,N-1.
        #reference trajectory for current robot.
        ref = cs.SX.sym('p',nx*(N+1))
        
        #other robots trajecctories including x and y for N steps
        c = cs.SX.sym('c',2*N*(self.nr_of_robots-1))

        # Parameters for 5 obstacles, 12 each
        o = cs.SX.sym('o',5*12)

        # Optimization variables, nu control inputs for N steps for one robot
        u = cs.SX.sym('u',nu*N)

        # extract current state values
        x, y, theta = ref[0], ref[1], ref[2]

        # Get weights from input vector as the last elements
        q,qtheta,r,qN,qthetaN,qobs,qaccV,qaccW,qpol = Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8]

        # Define the cost
        cost = 0

        # Cost for initial acceleration from previous state
        u0 = ref[3:5] 
        cost += self.cost_acceleration(u[:2],u0, qaccV, qaccW)

        # Cost for all acceleration 
        cost += self.cost_all_acceleration(u,qaccV,qaccW,N)
        
        # i: state index, j: control signal index, k: coordinate index
        for i,j,k in zip( range(0,nx*N,nx), range(0,nu*N,nu), range(0,2*N,2)):
            # Get the data for the current steps
            refi = ref[i:i+nx]
            xref, yref, thetaref= refi[0], refi[1], refi[2]
            uref = refi[3:]
            uj = u[j:j+nu]

            # Calculate the cost of all robots deviating from their reference
            #cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,q,qtheta)
            cost += q*self.cost_lines(ref,x,y,N)
            cost += self.cost_turn_left(theta,thetaref,qtheta)
            
            # Calculate the cost on all control actions
            cost += r*cs.dot(uref-uj,uref-uj)

            # Update the states
            x,y,theta = model(x,y,theta,uj,ts)

            # Cost of being inside an object 
            cost += self.cost_inside_polygon(x,y, o, qpol)
            
            # Avoid collisions
            #cost for dist to all other robots
            cost += self.cost_robot_collision(x,y, c, k, N, qobs)
            

        # Get the data for the last step
        refi = ref[nx*N:]
        xref, yref, thetaref= refi[0], refi[1], refi[2]

        # Calculate the cost of deviating from final reference
        cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=0.0,vmax=1.5,wmin=-1,wmax=1,N=N)
        
        # Concate all parameters
        p = cs.vertcat(ref,Q,c,o)
       

        return u,p,cost,bounds

    def build_mpc(self): 
        u,p,cost,bounds = self.generate_mpc_formulation()
       
        problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

        build_config = og.config.BuildConfiguration()\
            .with_build_directory("distributed1")\
            .with_build_mode("debug")\
            .with_tcp_interface_config()

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("distributed_solver_{}".format(self.nr_of_robots))

        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-4)\
            .with_max_duration_micros(50000)\
            .with_max_outer_iterations(15)
        
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                meta,
                                                build_config,
                                                solver_config)
        builder.build()
       

if __name__=='__main__':
    mpc = MPCGenerator(nr_of_robots=2)
    mpc.build_mpc()