import numpy as np
import matplotlib.pyplot as plt

def Koshi_manually(x):
    C1 = (733*np.sqrt(5) + 1827)/3606
    C2 = (609/1202 - 733*np.sqrt(5) / 3606)
    return C1 * np.exp((-8 + 4*np.sqrt(5))*x) + C2 * np.exp((-8 - 4*np.sqrt(5))*x) + np.exp(x) * (-8/601*np.cos(4*x) - 5/1803*np.sin(4*x))

x = np.arange(0, 2 + 0.001, 0.001)
plt.grid(True)
plt.plot(x, Koshi_manually(x))
