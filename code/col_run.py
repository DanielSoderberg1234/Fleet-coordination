import sys
import numpy as np
import matplotlib.pyplot as plt
import casadi.casadi as cs
import opengen as og
from function_lib import model


class MPCTester: 
    def __init__(self): 
        self.nu = 2
        self.N = 20
        self.ts = 0.1
        
    def get_input(self): 
        x,y,theta = [-0.5,0,0]
        q,qtheta,r,qN,qthetaN,qobs = 1,0.1,0.1,10,1,100
        p = [x,y,theta,q,qtheta,r,qN,qthetaN,qobs]
        return p

    def call_mpc(self, mpc_input):
        # Use TCP server
        # ------------------------------------
        mng = og.tcp.OptimizerTcpManager('collision/version1')
        mng.start()

        mng.ping()
        solution = mng.call(p=mpc_input, initial_guess=[1.0] * (self.nu*self.N))
        mng.kill()

        u_star = solution['solution']
        return u_star

    def plot_result(self, u_star, x,y,theta): 
        v = u_star[0::2]
        w = u_star[1::2]


        xlist = []
        ylist = []

        xlist.append(x)
        ylist.append(y)

        for vi,wi in zip(v,w): 
            x,y,theta = model(x,y,theta,[vi,wi],self.ts)
            xlist.append(x)
            ylist.append(y)

        plt.subplot(3,1,1)
        plt.plot(range(0,self.N),v)

        plt.subplot(3,1,2)
        plt.plot(range(0,self.N), w)

        plt.subplot(3,1,3)
        plt.plot(xlist,ylist,'-o')
        ang = np.linspace(0, 2*np.pi,100)
        r=0.25
        plt.plot(r*np.cos(ang), r*np.sin(ang) )

        plt.show()


if __name__=='__main__':
    mpctest = MPCTester()
    mpc_input = mpctest.get_input()
    u_star = mpctest.call_mpc(mpc_input)
    mpctest.plot_result(u_star, mpc_input[0], mpc_input[1],mpc_input[2])