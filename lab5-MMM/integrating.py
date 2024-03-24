import numpy as np
eps = 10**(-4)
a = 0
b = 9
h = (b - a) / 10000

def func(x):
    return (np.sin(x) * np.log(1 + x)) / (x*x * np.sqrt(x))

ans = []
x0 = a + h / 2
y0 = func(x0)
ans.append(y0)
for i in range(10000-1):
    xi = x0 + h
    yi = func(xi)
    ans.append(yi)
    x0 = xi
print(round(h*sum(ans), 5))

# import numpy as np
# eps = 10**(-4)
# a = 2
# b = 5
# h = (b - a) / 5

# def func(x):
#     return (1 / np.log(x))

# ans = []
# x0 = a + h / 2
# y0 = func(x0)
# ans.append(y0)
# print(x0)
# for i in range(5-1):
#     xi = x0 + h
#     yi = func(xi)
#     print(xi)
#     ans.append(yi)
#     x0 = xi
# print(h*sum(ans))