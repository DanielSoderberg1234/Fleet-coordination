import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np

x1 = 1
y1 = 2
ang = np.linspace(0,2*np.pi,100)

plt.plot(x1+np.cos(ang), y1+np.sin(ang))
plt.show()