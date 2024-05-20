import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def f(x):
    return np.sin(4*x) * np.exp(x)

def adams(f, N):
    a, b = 0, 2
    h = (b - a) / N
    x = np.arange(a, b + h, h)
    y = np.zeros(N+1)
    y[0] = 1
    y1 = np.zeros(N+1)
    y1[0] = 0
    
    for i in range(1, 2):
        k1 = h * y1[i-1]
        l1 = h * (f(x[i-1]) - 16*y1[i-1] + 16*y[i-1])
        k2 = h * (y1[i-1] + l1)
        l2 = h * (f(x[i-1]+h) - 16*(y1[i-1]+k1) + 16*(y[i-1]+k2))
        y[i] = y[i-1] + (k1 + k2) / 2
        y1[i] = y1[i-1] + (l1 + l2) / 2

    for i in range(2, N+1):
        y_pred = y[i-1] + h * (1.5 * y1[i-1] - 0.5 * y1[i-2])
        y1_pred = y1[i-1] + h * (1.5 * (f(x[i-1]) - 16*y1[i-1] + 16*y[i-1]) - 0.5 * (f(x[i-2]) - 16*y1[i-2] + 16*y[i-2]))
        y[i] = y[i-1] + h * (1.5 * y1_pred - 0.5 * y1[i-1])  
        y1[i] = y1[i-1] + h * (1.5 * (f(x[i-1]) - 16*y1_pred + 16*y_pred) - 0.5 * (f(x[i-2]) - 16*y1[i-2] + 16*y[i-2]))

    return [x, y, y1, h]


N = 10
eps = 0.01
count = 0
while True:
    count += 1
    A = adams(f, N)
    A2 = adams(f, 2*N)
    err = (norm(A[1] - A2[1][0::2]) + norm(A[2] - A2[2][0::2]))/(norm(A2[1]) + norm(A2[2]))
    N *=2
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
plt.plot(X[1:], Y, label = "Численное")
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.title('График решения y(x)')
plt.show()

#y'(x)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(X[1:] ,Y1, label = "Численное")
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
