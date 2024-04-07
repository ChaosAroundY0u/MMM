import numpy as np
# import scipy as sc

n, ai, ci, bi, pi = 99, 1, 1, 10, 1

def fill_array(N):
    array = [[0 for i in range(N)] for j in range(N)]
    
    for i in range(2, N):
        array[i][i-2] = ai
        array[i][i-1] = bi
        array[i][i] = ci
    
    return (array[2:])
# A = (([[bi, ci] + [0 for _ in range(n+1-2)]]))
# A_part = fill_array(n+1)
# print(A)
# A = np.array(A + A_part)
# print(A)

# A = np.array([[ai, bi, ci] + [0 for j in range(n-3)] for i in range(n+1)] , float) + np.array([1 for k in range(n+1)], float)
# A = np.array((np.array([bi, ci] + [0 for j in range(n+1-2)]) + np.array([[ai, bi, ci] + [0 for i in range(n+1-3)] for i in range(n)]) + np.array([pi for k in range(n+1)])), float )
A = (([[bi, ci] + [0 for _ in range(n+1-2)]]))

# A = np.vstack([A, (np.array([[ai, bi, ci] + [0 for i in range(n+1-3)] for j in range(n-1)]))])
A_part = fill_array(n+1)
A = np.array(A + A_part)

A = np.vstack([A, np.array([pi for k in range(n+1)])])
A = A.astype(float)
print(A)
#shape[0] - rows
#shape[1] - columns
print(A.shape[0], A.shape[1])

B = np.array([i+1 for i in range(n+1)], float).reshape(n + 1, 1)
print(B)
print(B.shape[0], B.shape[1])

#Gauss method
def gauss(A, B):
    n = len(B)
    m = n - 1
    i = 0
    x = np.zeros(n)   
    augmented_matrix = np.concatenate((A, B), axis = 1)
    while i < n:
        if augmented_matrix[i][i] == 0.0:
            print("Dividing by 0 WTF!!!")
            return
        for j in range(i + 1, n):
            coef = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (coef * augmented_matrix[i])
            
        i += 1
    
    x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]
    
    for k in range(n-2, -1, -1):
        x[k] = augmented_matrix[k][n]
        
        for j in range(k + 1, n):
            x[k] = x[k] - augmented_matrix[k][j] * x[j]
        x[k] = x[k] / augmented_matrix[k][k]
    # for ans in range(n):
    #     print("x", ans+1, "=", x[ans])
    
    return x

print("Solution with Gauss method: \n", gauss(A, B))

#Jacobi iteration method
def Jacobi(A, B, n = 200, init_guess = None):
    if init_guess is None:
        init_guess = np.zeros(len(A[0]))
    D = np.diag(A)
    T = A - np.diagflat(D)
    
    for i in range(n):
        init_guess = (B - np.dot(T, init_guess)) / D
        
    return init_guess
        
print("Solution with Jacobi method: \n", Jacobi(A, B))

#Eigenvalues
print("Lambda max = ", max(np.linalg.eigvals(A)), 
      "Lambda min = ", min(np.linalg.eigvals(A)))

#Condition number
print("1 Norm: ", np.linalg.cond(A, 1))
print("2 Norm: ", np.linalg.cond(A, 2))
print("frobenius norm: ",np.linalg.cond(A, 'fro'))
print("frobenius norm: ", np.linalg.norm(A, 'fro') * np.linalg.norm(np.linalg.inv(A), 'fro'))

#residual vector
rv_gauss = B - np.dot(A, gauss(A, B))
rv_jacobi = B - np.dot(A, Jacobi(A, B))
print("Residual vector for Gauss method: \n", rv_gauss)
print("Residual vector for Jacobi method: \n", rv_jacobi)
print("Norm of residual vector for Gauss method: ", np.sqrt(np.sum(rv_gauss**2)))
print("Norm of residual vector for Jacobi method: ", np.sqrt(np.sum(rv_jacobi**2)))