import casadi.casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block 

do = {'center': [1,1], 'a': 2, 'b': 1, 'vel': [1,0]}

def predict(do, N):
        xy = do['center']
        for i in range(0,N): 
                xy.append( xy[-2]+do['vel'][0])
                xy.append( xy[-2]+do['vel'][1])
        return xy



def plot_ellipses(do, centers): 
        ang = np.linspace(0,2*np.pi, 100)
        for i in range(0, len(centers), 2): 
                c = centers[i:i+2]
                x = c[0] + do['a']*np.cos(ang)
                y = c[1] + do['b']*np.sin(ang)

                if i == 0: 
                        plt.plot(x,y,color='b')
                else: 
                        plt.plot(x,y,color='b', alpha= 0.2)

plot_ellipses(do, predict(do,20))
plt.show()

