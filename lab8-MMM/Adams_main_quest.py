import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def f(x):
    return np.sin(4*x) * np.exp(x)

def adams(N):
    a, b = 0, 2
    h = 2 / N
    x = np.arange(a, b, h)
    y = np.zeros(N+1)
    y[0] = 1
    y1 = np.zeros(N+1)
    y1[0] = 0
    
    y2 = f(x[0]) - 16*y1[0] - 16*y[0]
    for i in range(1, N):
        y[i+1] = y[i] + h*y1[i]
        y1[i+1] = y1[i] + h*y2
        y2 = f(x[i]) - 16*y1[i] - 16*y[i]
    return [x, y, y1, h]


N = 10
eps = 0.01
count = 0
while True:
    count += 1
    A = adams(N)
    A2 = adams(2*N)
    err = (norm(A[1] - A2[1][0::2]) + norm(A[2] - A2[2][0::2]))/(norm(A2[1]) + norm(A2[2]))
    N *= 2
    if err < eps:
        print("Погрешность:", err)
        break
X, Y, Y1, h = A2
Y = Y[1:]
Y1 = Y1[1:]
print("Итераций:",count)
print("Шаг:", A2[3])
#y(x)
plt.figure(figsize=(10,4))
plt.plot(X,Y, label = "Численное")
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.title('График решения y(x)')
plt.show()

#y'(x)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(X,Y1, label = "Численное")
plt.grid(True)
plt.xlabel('x')
plt.ylabel("y'")
plt.legend(loc='upper right')
plt.title("График решения y'(x)")
plt.subplot(1,2,2)
plt.grid(True)
plt.xlabel('x')
plt.title('Разностный график')
plt.show()

#y'(y)
plt.plot(Y,Y1, label = "Численное")
plt.grid(True)
plt.xlabel("y")
plt.ylabel("y'")
plt.title("Фазовая траектория y'(y)")
plt.show()
