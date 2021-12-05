import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model
from itertools import combinations

"""
    A file that generates the MPC formulation.
"""
class Centralized: 
    def __init__(self, nr_of_robots): 
        self.name = "Fleet-collison"
        self.nr_of_robots = nr_of_robots

    def shortest_dist_to_ref(self,ref,x,y,N):
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

    def cost_line(self,robots,qdist): 
        nu = 2
        nx = 5
        N = 20

        cost = 0

        # Loop over all robots 
        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            ref = robots[robot_id]['Ref']

            shortest_dist = self.shortest_dist_to_ref(ref,x,y,N)
            cost += shortest_dist*qdist 

        return cost   


    def cost_state_ref(self,x,y,theta,xref,yref,thetaref,q,qtheta): 
        # Cost for deviating from the current reference
        return q*( (xref-x)**2 + (yref-y)**2 ) + qtheta*(thetaref-theta)**2

    def cost_deviation_ref(self,robots,i,j,q,qtheta):
        nu = 2
        nx = 5
        cost = 0

        # Loop over all robots 
        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            ref = robots[robot_id]['Ref']

            # Get the data for the current steps
            refi = ref[i:i+nx]

            # Get the references explicit
            xref, yref, thetaref = refi[0], refi[1], refi[2]
            cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,q,qtheta)
        
        return cost

    def update_robot_states(self,robots,i,j,ts): 
        nu = 2
        nx = 5

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            uj = u[j:j+nu]
            x,y,theta = model(x,y,theta,uj,ts)
            robots[robot_id]['State'] = [x,y,theta]


    def cost_robot2robot_dist(self,x1,y1,x2,y2,qobs): 
        # Cost for being closer than r to the other robot
        return qobs*cs.fmax(0.0, 1.0**2 - (x1-x2)**2 - (y1-y2)**2)**2

    def cost_collision(self,robots, qobs):
        cost = 0 
        # Iterate over all pairs of robots
        for comb in combinations(range(0,self.nr_of_robots),2):
            x1,y1,theta1 = robots[comb[0]]['State']
            x2,y2,theta2 = robots[comb[1]]['State']
            cost += self.cost_robot2robot_dist(x1,y1,x2,y2,qobs)

        return cost
        
    def cost_control_action(self,u,u_ref,r): 
        # Cost for the control action
        return r*cs.dot(u_ref-u,u_ref-u)

    def cost_all_control_action(self,robots,i,j,r): 
        nu = 2
        nx = 5
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']
            refi = robots[robot_id]['Ref'][i:i+nx]
            u_ref = refi[3:]
            uj = u[j:j+nu]
            cost += self.cost_control_action(uj,u_ref,r)
        
        return cost

    def cost_acceleration(self,u0,u1,qaccV,qaccW): 
        cost = 0
        cost += qaccV*(u1[0]-u0[0])**2
        cost += qaccW*(u1[1]-u0[1])**2
        return cost
        

    def cost_all_acceleration(self,robots,qaccV,qaccW): 
        nu = 2
        nx = 5
        N = 20
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            x,y,theta = robots[robot_id]['State']
            u = robots[robot_id]['u']

            u0 = u[:-2]
            u1 = u[2:]

            for i in range(0,N-2,2):
                u0i = u0[i:i+2]
                u1i = u1[i:i+2]
                cost += self.cost_acceleration(u0i,u1i,qaccV,qaccW)
            
        return cost

    def cost_inital_acceleration(self,robots,qaccV,qaccW): 
        nu = 2
        nx = 5
        N = 20
        cost = 0

        for robot_id in robots: 
            # Extract the content for each robot
            v_ref,w_ref = robots[robot_id]['Init_u']
            u_ref = cs.vertcat(v_ref,w_ref)
            u = robots[robot_id]['u'][:nu]
            cost += self.cost_acceleration(u,u_ref,qaccV,qaccW)
            
        return cost

    def cost_inside_polygon(self,robots,o,qobs): 
        cost = 0.0
        for robot_id in robots: 
            x,y,theta = robots[robot_id]['State']

            for i in range(0,5): 
                # Parameter for each object
                ob = o[i*12:(i+1)*12]

                inside = 1
                for j in range(0,12,3):
                    h = ob[j:j+3]
                    inside *= cs.fmax(0.0, h[2] - h[1]*y - h[0]*x )**2

                cost += qobs*inside

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
        N = 20
        for robot_id in robots: 
            x,y,theta = robots[robot_id]['State']
            a,b, phi = e[0],e[1],e[2]
            centers = e[3:]

            # Equation for ellipse from: https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
            c = centers[j:j+2]
            cost += qpol*cs.fmax(0.0, 1.0 - ((x-c[0])*cs.cos(phi)+(y-c[1])*cs.sin(phi))**2/a**2 - ((x-c[0])*cs.sin(phi)-(y-c[1])*cs.cos(phi))**2/b**2)

        return cost

    def bound_control_action(self, vmin,vmax,wmin,wmax,N): 
        # But hard constraints on the velocities of the robot
        N = self.nr_of_robots*N
        umin = [vmin,wmin]*N
        umax = [vmax,wmax]*N
        return og.constraints.Rectangle(umin, umax)

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 5, 20, 0.1)

        # Input vector nr_of_robot trajectories, N+1 long with nx states in each i=0,1,2,..,N
        p = cs.SX.sym('p',self.nr_of_robots*nx*(N+1))

        # Number of weights 
        Q = cs.SX.sym('Q',11)

        # Parameters for 5 obstacles, 12 each
        o = cs.SX.sym('o',5*12)

        # Parameters for boundaries, 12 
        b = cs.SX.sym('b',12)

        # Parameters for a dynamic obstacle, described by an ellipse with a and b and then N centers
        e = cs.SX.sym('e',3+N*2)

        # Optimization variables 2 robots each with nu control inputs for N steps
        u = cs.SX.sym('u',self.nr_of_robots*nu*N)

        # Dictionary to hold all robot data
        robots = {}

        # Fill the dictionary
        for i in range(0,self.nr_of_robots): 
            # Values to fill the dictionary
            ref_r = p[(N+1)*nx*i:(N+1)*nx*(i+1)]
            u_r = u[nu*N*i:nu*N*(i+1)]
            x_r,y_r,theta_r,v_ref,w_ref = ref_r[0], ref_r[1], ref_r[2], ref_r[3], ref_r[4]

            # All data for robot i, current state, control inputs for all states, reference for all states
            robots[i] = {"State": [x_r,y_r,theta_r], "Init_u": [v_ref,w_ref], 'u': u_r, 'Ref': ref_r}

        
        # Get weights from input vectir as the last elements
        q, qtheta, r, qN, qthetaN,qobs, qaccV,qaccW, qpol, qbound, qdyn = Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9],Q[10]

        # Define the cost
        cost = 0
        
        # Penalize the first acceleration
        cost += self.cost_inital_acceleration(robots,qaccV,qaccW)

        for i,j in zip( range(0,nx*N,nx), range(0,nu*N,nu)): 
            # Calculate the cost of all robots deviating from their reference
            cost += self.cost_deviation_ref(robots,i,j,q,qtheta)
            #cost += self.cost_line(robots,q)
            # Calculate the cost on all control actions
            cost += self.cost_all_control_action(robots,i,j,r)
            # Update the states
            self.update_robot_states(robots,i,j,ts)
            # Calculate the cost of colliding
            cost += self.cost_collision(robots, qobs)  
            # Cost of being outside boundaries
            cost += self.cost_outside_boundaries(robots,b,qbound)
            # Cost of being inside an object 
            cost += self.cost_inside_polygon(robots, o, qpol)
            # Cost for dynamic obstavle
            cost += self.cost_dynamic_obstacle(robots,e,j,qdyn)
            
        # Add acceleration cost
        cost += self.cost_all_acceleration(robots,qaccV,qaccW)

        # Cost for deviating from final reference points
        cost += self.cost_deviation_ref(robots,nx*N,nu*N,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=.0,vmax=1.5,wmin=-1,wmax=1,N=N)
        
        # Concate all parameters
        p = cs.vertcat(p,Q,o,b,e)

        return u,p,cost,bounds

    def build_mpc(self): 
        u,p,cost,bounds = self.generate_mpc_formulation()
       
        problem = og.builder.Problem(u, p, cost)\
            .with_constraints(bounds) \
                  

        build_config = og.config.BuildConfiguration()\
            .with_build_directory("collision_avoidance")\
            .with_build_mode("debug")\
            .with_tcp_interface_config()

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("robot_{}_solver".format(self.nr_of_robots))

        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-4)\
            .with_max_duration_micros(50000)\
            .with_max_outer_iterations(15)
        
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                meta,
                                                build_config,
                                                solver_config)
        builder.build()

class Distributed: 
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
    
    def cost_collision(self, x, y, c, other_robots, k, N, qobs):
        cost = 0.0
        for other_robot_nr in range(self.nr_of_robots-1):
            #2 = nr_of_coord (x,y),                 
            ck = c[2*N*other_robot_nr + k : 2*N*other_robot_nr + k + 2]
            xc,yc = ck[0],ck[1]
            cost += other_robots[other_robot_nr]*self.cost_robot2robot_dist(x,y,xc,yc,qobs)
        return cost


    def cost_outside_boundaries(self,x, y,b,qb): 
        cost = 0.0
        outside = 0
        for j in range(0,12,3):
            h = b[j:j+3]
            outside += cs.fmin(0.0, h[2] - h[1]*y - h[0]*x )**2

        cost += qb*outside

        return cost

    def cost_dynamic_obstacle(self, x, y, e, j, qdyn): 
        cost = 0.0 
        
        a,b, phi = e[0],e[1],e[2]
        centers = e[3:]

        # Equation for ellipse from: https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
        c = centers[j:j+2]
        cost += qdyn*cs.fmax(0.0, 1.0 - ((x-c[0])*cs.cos(phi)+(y-c[1])*cs.sin(phi))**2/a**2 - ((x-c[0])*cs.sin(phi)-(y-c[1])*cs.cos(phi))**2/b**2)

        return cost

    def generate_mpc_formulation(self): 

        # Some predefined values, should maybe be read from a config file?
        (nu, nx, N, ts) = (2, 5, 20, 0.1)

        # the 8 last are the weights
        Q = cs.SX.sym('Q',11)

        # Input vector 1 trajectory, N long with nx states in each i=0,1,2,..,N-1.
        #reference trajectory for current robot.
        ref = cs.SX.sym('p',nx*(N+1))
        
        #other robots trajecctories including x and y for N steps
        c = cs.SX.sym('c',2*N*(self.nr_of_robots-1))

        # Parameters for 5 obstacles, 12 each
        o = cs.SX.sym('o',5*12)

        # Parameters for boundaries, 12 
        b = cs.SX.sym('b',12)

        # Parameters for a dynamic obstacle, described by an ellipse with a and b and then N centers
        e = cs.SX.sym('e',3+N*2)

        # Parameter for number of other robot trajectories to consider
        other_robots = cs.SX.sym('other_robots',self.nr_of_robots-1)

        # Optimization variables, nu control inputs for N steps for one robot
        u = cs.SX.sym('u',nu*N)

        # extract current state values
        x, y, theta = ref[0], ref[1], ref[2]

        # Get weights from input vector as the last elements
        q, qtheta, r, qN, qthetaN,qobs, qaccV,qaccW, qpol, qbounds, qdyn = Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9],Q[10]

        # Define the cost
        cost = 0

        # Cost for initial acceleration from previous state
        u0 = ref[3:5] 
        cost += self.cost_acceleration(u[:2],u0, qaccV, qaccW)
        
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
            # cost for turning left
            cost += self.cost_turn_left(theta,thetaref,qtheta)
            # Calculate the cost on all control actions
            cost += r*cs.dot(uref-uj,uref-uj)
            # Update the states
            x,y,theta = model(x,y,theta,uj,ts)
            #cost for dist to all other robots
            cost += self.cost_collision(x,y, c, other_robots, k, N, qobs)
            # Cost of being inside an object 
            cost += self.cost_outside_boundaries(x,y, b, qbounds)
            # Cost of being inside an object 
            cost += self.cost_inside_polygon(x,y, o, qpol)
            # Cost for dynamic obstacle
            cost += self.cost_dynamic_obstacle(x,y,e,j,qdyn)

        # Cost for all acceleration 
        cost += self.cost_all_acceleration(u,qaccV,qaccW,N)

        # Get the data for the last step
        refi = ref[nx*N:]
        xref, yref, thetaref= refi[0], refi[1], refi[2]

        # Calculate the cost of deviating from final reference
        cost += self.cost_state_ref(x,y,theta,xref,yref,thetaref,qN,qthetaN)

        # Get the bounds for the control action
        bounds = self.bound_control_action(vmin=0.0,vmax=1.5,wmin=-1,wmax=1,N=N)
        
        # Concate all parameters
        p = cs.vertcat(ref,Q,c,other_robots,o,b,e)

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
       
       

def main(): 
    #cen = Centralized(nr_of_robots=2)
    #cen.build_mpc()

    dis = Distributed(nr_of_robots=20)
    dis.build_mpc()


if __name__=='__main__':
    main()
    