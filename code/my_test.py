import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# model function
def model(x, y, theta, v, w, ts):

    x += ts*cs.cos(theta)*v
    y += ts*cs.sin(theta)*v
    theta += ts*w
    return x, y, theta

# reference trajectory generator
def generate_straight_line(x, y, theta, v, ts, N):
    
    states = [x, y, theta]
    
    for i in range(0,N):
        x,y,theta = model(x, y, theta, v, 0, ts)
        states.extend([x,y,theta])

    return states


# define system parameters
# nnumber of control signals, system states, horizon, sample time
(nu, nx, N, ts) = (2, 3, 20, 0.1)

# Penalties. Q: x y, Qtheta: theta, R: Control signals, QN: final x y, QthetaN: final theta
(Q, Qtheta, R, QN, QthetaN, Qdist) = (1, 1, 1, 1, 1, 1)

# define symbolic variables to be optimized
# u: nu number of controls for horizon N
# p: nx number of states and references
"""u1 = cs.SX.sym('u1', nu*N)
u2 = cs.SX.sym('u2', nu*N)
p1 = cs.SX.sym('p1', nx*2)
p2 = cs.SX.sym('p2', nx*2)"""

u = cs.SX.sym('u', 2*nu*N)
z0 = cs.SX.sym('z0', 2*nx)

u1 = u[:nu*N]
u2 = u[nu*N:]

"""p = cs.SX.sym('p', 2*nx*N)"""

# extract initial states for x,y, and theta
(x1, y1, theta1) = (z0[0], z0[1], z0[2])
(x2, y2, theta2) = (z0[3], z0[4], z0[5])

"""(x1, y1, theta1) = (p[0], p[1], p[2])
(x2, y2, theta2) = (p[nx*(N+1)], p[nx*(N+1)+1], p[nx*(N+1)+2])
"""
# create dummy trajectories
# robot 1
ref_traj1 = generate_straight_line(-1, 0, 0, 1, 0.1, 20)
# robot 2
ref_traj2 = generate_straight_line(0, -1, cs.pi/2, 1, 0.1, 20)

# initial cost to zero
cost = 0

xyref1 = np.zeros(2*N)
xyref2 = np.zeros(2*N)

# loop from zero to nu*N, nmbr of controls times the horizon, intervals is number of controls
for t in range(0, N):
    
    # extract reference trajectories for current time step 
    x_ref1, y_ref1, theta_ref1 = ref_traj1[t*nx:(t+1)*nx]
    x_ref2, y_ref2, theta_ref2 = ref_traj2[t*nx:(t+1)*nx]

    xyref1[t] = x_ref1
    xyref1[t+N] = y_ref1
    xyref2[t] = x_ref2
    xyref2[t+N] = y_ref2
    
    # compute and add cost for current deviation from reference.
    cost += Q*((x1 - x_ref1)**2 + (y1 - y_ref1)**2) + Qtheta*(theta1 - theta_ref1)**2
    cost += Q*((x2 - x_ref2)**2 + (y2 - y_ref2)**2) + Qtheta*(theta2 - theta_ref2)**2

    # extract controls for current time step
    u_t1 = u1[t*nu : (t+1)*nu]
    u_t2 = u2[t*nu : (t+1)*nu]

    # predict next states
    states1 = model(x1, y1, theta1, u_t1[0], u_t1[1], ts)
    states2 = model(x2, y2, theta2, u_t2[0], u_t2[1], ts)

    cost += R*cs.dot(u_t1, u_t1)
    cost += R*cs.dot(u_t2, u_t2)

    (x1, y1, theta1) = (states1[0], states1[1], states1[2])
    (x2, y2, theta2) = (states2[0], states2[1], states2[2])

    cost += Qdist*(cs.fmax(0.0, 1**2 - (x1 - x2)**2 - (y1 - y2)**2))


# compute and add cost for final state
cost += QN*((x1 - x_ref1)**2 + (y1 - y_ref1)**2) + Qtheta*(theta1 - theta_ref1)**2
cost += QN*((x2 - x_ref2)**2 + (y2 - y_ref2)**2) + Qtheta*(theta2 - theta_ref2)**2

# Control signal bounds umin = [vmin, wmin], umax = [vmax, vmin]
umin = [-3.0, -1.0]*(nu*N)
umax = [3.0, 1.0]*(nu*N)

# define bounds for control signal
bounds = og.constraints.Rectangle(umin, umax)

bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, z0, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("python_test_build")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()

mng = og.tcp.OptimizerTcpManager('python_test_build/navigation')
mng.start()

(x01, y01, theta01) = (-1, 0, 0)
(x02, y02, theta02) = (0, -1, cs.pi/2)

mng.ping()
solution = mng.call([x01, y01, theta01, x02, y02, theta02], initial_guess=[1.0, 1.0] * (nu*N))
mng.kill()

time = np.arange(0, ts*N, ts)
u_star = solution['solution']
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]

plt.subplot(221)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(222)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.xlabel('Time')

plt.subplot(223)

plt.plot(xyref1[:N], xyref1[N:], '-o')
plt.plot(xyref2[:N], xyref2[N:], '-o')
plt.show()






