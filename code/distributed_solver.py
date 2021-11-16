import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from function_lib import model
from function_lib import generate_straight_trajectory

nr_robots = 2
(nu, nx, N, ts) = (2, 3, 20, 0.1)

(q, qtheta, r, qN, qthetaN, qd) = (10, 100, 1, 100, 100, 1000)

u = cs.SX.sym('u', nu*N)
p = cs.SX.sym('p', (nx-1)*(nr_robots-1) + nx*(N+1))

print("p length: {}".format((nx-1)*(nr_robots-1) + nx*(N+1)))

(x, y, theta) = (p[0], p[1], p[2])

cost = 0
for iu in range(0, nu*N, nu):
    
    ix = nx*iu/2
    
    xref = p[nx + ix]
    yref = p[nx + ix + 1]
    thetaref = p[nx + ix + 2]

    u_t = u[iu : iu + nu]

    xobs = p[-8]
    yobs = p[-7]

    cost += q*((x - xref)**2 + (y - yref)**2) + qtheta*(theta - thetaref)**2
    
    cost += r * cs.dot(u_t, u_t)

    (x, y, theta) = model(x, y, theta, u_t, ts)

    cost += qd*cs.fmax(0.0, 1.0**2 - (x - xobs)**2 - (y - yobs)**2)
    

cost += qN*((x-xref)**2 + (y-yref)**2) + qthetaN*(theta-thetaref)**2

umin = [-3.0] * (nu*N)
umax = [3.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, p, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("dist_collision_avoidance")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("dist_solver")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()










