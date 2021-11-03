import casadi.casadi as cs
import numpy as np 
import matplotlib.pyplot as plt

x = []
y = []


for i in range(0,3):
    x.append(i)
    y.append(i)

    plt.plot(x,y,'-o','r')
    plt.pause(1)

plt.xaxis([-5,5])
plt.yaxis([-5,5])
plt.show()