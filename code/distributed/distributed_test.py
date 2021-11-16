import matplotlib.pyplot as plt 
import numpy as np 
import casadi.casadi as cs
from function_lib import model, generate_straight_trajectory
import opengen as og
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from time import perf_counter_ns
from RobotModelData import RobotModelData

# System parameters
(nx, nu, N, ts) = (3, 2, 20, 0.1)

ref1 = generate_straight_trajectory(-1, 0, 0, 1, 0.1, 20)
ref2 = generate_straight_trajectory(0, -1, cs.pi/2, 1, 0.1, 20)

# Extend trajectory with final state 50 times
final_state1 = ref1[-3:]
final_state2 = ref2[-3:]
print("Final state 1: {}".format(final_state1))
for i in range(50):
    ref1.extend(final_state1)
    ref2.extend(final_state2)


u_p1 = [0,0]
u_p2 = [0,0]

x1, y1, theta1 = ref1[:nx]
x2, y2, theta2 = ref2[:nx]

x_p1, y_p1 = x1, y1
x_p2, y_p2 = x2, y2

X1 = []
Y1 = []
X2 = []
Y2 = []
X1.append(x1)
Y1.append(y1)
X2.append(x2)
Y2.append(y2)

# Distributed algorithm parameters
(Pmax, P, eps, K, w) = (100, 1, 0.1, 10, 0.5)

mng = og.tcp.OptimizerTcpManager('dist_collision_avoidance/dist_solver')
mng.start()
mng.ping()

for t in range(N):
    
    # initialize K, P
    K = 10
    P = 1

    print("t: {}".format(t))
    while P <= Pmax and K > eps:
        
        print("P iter: {} ".format(P))
        
        # initialize solver inputs
        solver_input_1 = []
        solver_input_2 = []
        # insert reference trajectories to solver inputs
        solver_input_1.extend(ref1[t*nx: (N+t+1)*nx])
        solver_input_2.extend(ref2[t*nx: (N+t+1)*nx])
        # insert predicted position for each of the 2 robots
        solver_input_1.extend([x_p2, y_p2])
        solver_input_2.extend([x_p1, y_p1])
        
        print("input 1 length {}".format(len(solver_input_1)))

        u_p1_last = u_p1
        u_p2_last = u_p2

        # Robot 1
        response1 = mng.call(p=solver_input_1, initial_guess=[1.0] * (nu*N))
        
        if response1.is_ok():
            print("OK")
            solution_data1 = response1.get()
            u_star1 = solution_data1.solution[:2]
            exit_status1 = solution_data1.exit_status
            solver_time1 = solution_data1.solve_time_ms
        else:
            print("NOT OK")
            solver_error1 = response1.get()
            error_code1 = solver_error1.code
            error_msg1 = solver_error1.message
            print("Error code {}".format(error_code1))
            print(error_msg1)

        print("u_star 1: {}".format(u_star1))

        # compute prediction for current iteration
        u_p1 = [w*u_star1[0] + (1 - w)*u_p1[0], w*u_star1[1] + (1 - w)*u_p1[1]]
        # find max difference in predicted u from last prediction
        K1 = max([abs(u_p1[0]-u_p1_last[0]), abs(u_p1[1]-u_p1_last[1])])
        # predict next state
        x_p1, y_p1, theta_p1 = model(x1, y1, theta1, u_p1, ts)

        # Robot 2
        response2 = mng.call(p=solver_input_2, initial_guess=[1.0] * (nu*N))

        if response2.is_ok():
            solution_data2 = response2.get()
            u_star2 = solution_data2.solution[:2]
            exit_status2 = solution_data2.exit_status
            solver_time2 = solution_data2.solve_time_ms
        else:
            solver_error2 = response2.get()
            error_code2 = solver_error2.code
            error_msg2 = solver_error2.message

        u_p2 = [w*u_star2[0] + (1 - w)*u_p2[0], w*u_star2[1] + (1 - w)*u_p2[1]]
        K2 = max([abs(u_p2[0]-u_p2_last[0]), abs(u_p2[1]-u_p2_last[1])])

        x_p2, y_p2, theta_p2 = model(x2, y2, theta2, u_p2, ts)

        # find max of the control prediction differences from robot 1 and 2
        K = max(K1, K2)
        P += 1
        
    # take step in simulation
    x1, y1, theta1 = model(x_p1, y_p1, theta_p1, u_p1, ts)
    x2, y2, theta2 = model(x_p2, y_p2, theta_p2, u_p2, ts)
    
    # save states to print
    X1.append(x1)
    Y1.append(y1)
    X2.append(x2)
    Y2.append(y2)
    
mng.kill()

for t in range(N):
    plt.plot(X1[:t], Y1[:t], '-o', color = 'r')
    plt.plot(X2[:t], Y2[:t], '-o', color = 'b')
    plt.show(block = False)
    plt.pause(2)





