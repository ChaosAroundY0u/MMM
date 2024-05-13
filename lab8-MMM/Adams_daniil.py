import numpy as np
import matplotlib.pyplot as plt

def diffur(x, y):
    q, y2 = y
    y1, q = -16*y2 + 16*y + np.sin(4*x) * np.exp(x)
    return np.array([y2, y1])
y0 = np.array([1, 0])
x = np.arange(0, 2 + 0.001, 0.001)
y = np.zeros((len(x), len(y0)))

y[0] = y0

print(y[0], np.array(diffur(0, y[0])))

y[1] = y[0] + (0.001 * np.array(diffur(0, y[0])))

print(y[1])

print(np.array(diffur(0, y[0])), '*0.001 + 1 = ', y[0] + (0.001 * np.array(diffur(0, y[0]))))

for i in range(1, len(x) - 1):
    y[i+1] = y[i] + 0.001/2 * (3*np.array(diffur(x[i], y[i])) - np.array(diffur(x[i-1], y[i-1]))) #last pairs
    
print(y)

dy = np.gradient(y[:, 0], 0.001) #phase trajectory


plt.figure(1)
plt.plot(x, y[:, 0])
points = [0] * len(y)
points0 = [None] * len(x)
for i in range(len(y)):
    if i % 100 == 0:
        points[i], q = y[i]
        points0[i] = x[i]
plt.plot(points0, points, marker = ".")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('y(x)')
plt.show()

plt.figure(2)
plt.plot(x, dy)
points = [None] * len(dy)
points0 = [None] * len(x)
for i in range(len(dy)):
    if i % 100 == 0:
        points[i] = dy[i]
        points0[i] = x[i]
plt.plot(points0, points, marker = ".")
plt.xlabel('X')
plt.ylabel('dY/dX')
plt.title('Производная')
plt.show()
