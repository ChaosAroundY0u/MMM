import numpy as np
eps = 10**(-4)
a = 0
b = 9
start = 10

def func(x):
    return (np.sin(x) * np.log(1 + x)) / (x*x * np.sqrt(x))

def calculating(start):
    ans = []
    h = (b - a) / start
    x0 = a + h / 2
    y0 = func(x0)
    ans.append(y0)
    for i in range(start - 1):
        xi = x0 + h
        yi = func(xi)
        ans.append(yi)
        x0 = xi
    return h*sum(ans)
i = 1
k = 0
first_integral = 0
second_integral = calculating(start)
print("iteration ", k, ":", second_integral, "||", "pogr = ", second_integral - first_integral)

while abs(second_integral - first_integral) > eps:
    first_integral = second_integral
    i *= 2
    second_integral = calculating(start * i)
    k += 1
    print("iteration ", k, ":", second_integral, "||", "pogr = ", second_integral - first_integral)
