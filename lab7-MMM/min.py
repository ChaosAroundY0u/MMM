import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1, x2 = x
    return x1**2 - 22*x1 + 2*x2**2 + 28*x2 + 222

def gradf(x):
    x1, x2 = x
    return np.array([2*x1 - 22, 4*x2 + 28])

def gradient_method(x):
    a = 2
    lmbd = 0.1
    eps = 10**(-4)
    count = 0
    path = [x.copy()]
    while True:
        gradient = gradf(x)
        if np.linalg.norm(gradient) < eps:
            break
        while f(x - a * gradient) >= f(x): #беру "-" т.к. использую градиент, а не антиградиент
            a *= lmbd
        x = x - a * gradient
        count += 1
        path.append(x.copy())
    print(count)
    return x, path

init_guess = np.array([-11, 9])
min_point, path = gradient_method(init_guess)

x1 = np.linspace(-25, 25, 400)
x2 = np.linspace(-25, 25, 400)

X1, X2 = np.meshgrid(x1, x2)
F = f([X1, X2])

plt.contour(X1, X2, F, levels = 50)
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], marker = "o", color = "red")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print(f"Minimum point = {min_point}")
print(f"f(Minimum point) = {f(min_point)}")