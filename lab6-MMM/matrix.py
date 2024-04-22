import numpy as np
import random

n, ai, ci, bi, pi = 99, 1, 1, 10, 1

def fill_array(N):
    array = [[0 for i in range(N)] for j in range(N)]
    
    for i in range(2, N):
        array[i][i-2] = ai
        array[i][i-1] = bi
        array[i][i] = ci
    
    return (array[2:])

A = (([[bi, ci] + [0 for _ in range(n+1-2)]]))
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
    return x

print("Solution with Gauss method: \n", gauss(A, B))

#Jacobi iteration method
def Jacobi(A, B):
    x = np.array([0.1 for i in range(len(A[0]))])
    x0 = np.zeros_like(x)
    D = np.diag(A)
    count = 0
    R = A - np.diagflat(D)
    while(np.linalg.norm(x - x0) > 10**-4):
        x0 = x
        x = (B - np.dot(R, x0)) / D
        count += 1
    return x, count

B_J= np.array([i+1 for i in range(n+1)], float)
print("Solution with Jacobi method: \n", Jacobi(A, B_J))

#Eigenvalues
print("Lambda max = ", max(np.linalg.eigvals(A))) 
print("Lambda min = ", min(np.linalg.eigvals(A)))

#Condition number
print("1 Norm: ", np.linalg.cond(A, 1))
print("2 Norm: ", np.linalg.cond(A, 2))
print("frobenius norm: ",np.linalg.cond(A, 'fro'))
print("frobenius norm: ", np.linalg.norm(A, 'fro') * np.linalg.norm(np.linalg.inv(A), 'fro'))

rv_gauss = np.dot(A, gauss(A, B)) - np.array([i for i in range(1, 101)])

rv_jacobi = np.dot(A,Jacobi(A, B_J)[0]) - np.array([i for i in range(1, 101)])

print("Residual vector for Gauss method: \n", rv_gauss)
print("Residual vector for Jacobi method: \n", rv_jacobi)
print("Norm of residual vector for Gauss method: ", np.sqrt(np.sum(rv_gauss**2)))
print("Norm of residual vector for Jacobi method: ", np.sqrt(np.sum(rv_jacobi**2)))

def eigenvalues(A): #does not work !!
    n = A.shape[0]
    V = np.eye(n)
    count = 0
    while True:
        Q, R = np.linalg.qr(A)
        A_new = np.dot(R, Q)
        V = np.dot(V, Q)
        count += 1
        if np.abs(np.diag(A_new) - np.diag(np.diag(A_new))).max() < 10**(-4):
            break
        A = A_new
    eigenvals = np.diag(A_new)
    return eigenvals, count
print(eigenvalues((A)))
