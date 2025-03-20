import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

def read_data(file):
    A = []
    B = []
    with open(file, 'r') as f:
        for line in f:
            a, b = line.strip().split(",")
            A.append(float(a))
            B.append(float(b))
    return A, B

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]

    return L, U

def LU_inverse(L, U, b):
    n = A.shape[0]
    inv_A = np.zeros_like(A)
    identity_matrix = np.eye(n)
    # LUx = ATB  ->  L(Ux) = ATB ->  Ly = ATB, Ux = y
    # Ly = ATB
    n = L.shape[0]
    y = np.zeros_like(b)

    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum) / L[i][i]

    # Ux = y
    n = U.shape[0]
    x = np.zeros_like(y)

    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += U[i][j] * x[j]
        x[i] = (y[i] - sum) / U[i][i]

    return x
 

def LSE(A, B, lambd):
    A = np.array(A)
    B = np.array(B)
    I = np.eye(A.shape[1])
    ATA = A.T @ A + lambd * I
    ATB = A.T @ B
    # ATA = LU
    L, U = lu_decomposition(ATA)
    # LUx = ATB
    x = LU_inverse(L, U, ATB)
    return x

def compute_error(A, B, x):
    B_pred = A @ x
    error = np.sum((B - B_pred) ** 2)
    return error
    
def steepest_descent(A, B, lambd, learning_rate=0.00001, max_iter=10000):
    A = np.array(A)
    B = np.array(B)
    x = np.zeros(A.shape[1])
    I = np.eye(A.shape[1])
    for i in range(max_iter):
        # Gf with L1 regularization
        Gf = 2 * A.T @ A @ x - 2 * A.T @ B + lambd * np.sign(x)
        x = x - learning_rate * Gf
    return x

def newton_method(A, B, lambd):
    A = np.array(A)
    B = np.array(B)
    # initial guess
    x = np.zeros(A.shape[1]) 
    I = np.eye(A.shape[1])
    Gf = 2 * A.T @ A @ x - 2 * A.T @ B
    Hf = 2 * A.T @ A 
    # + lambd * np.eye(A.shape[1])
    # Bx = I -> LUx = I -> Ly = I, Ux = y 
    L, U = lu_decomposition(Hf)
    # x1 = x - Hf^-1 * Gf 
    x1 = x - LU_inverse(L, U, Gf)
    return x1


def plot_graph(A, B, x):
    B_pred = A @ x
    plt.scatter(data_a, data_b, label="Data")
    plt.plot(data_a, B_pred, label="Prediction")
    plt.legend()
    plt.show()

def show_result(x):
    if N == 2:
        print(f"Fitting line: {x[0]} x + {x[1]}")
    elif N == 3:
        print(f"Fitting line: {x[0]} x^2 + {x[1]} x + {x[2]}")
    error = compute_error(np.array(A), np.array(B), x)
    print(f"Total error: {error}")
    plot_graph(A, B, x)

# Read data
data_a, data_b = read_data("data.txt")
N = 3  # Degree of polynomial
lambd = 10000
A = []
for i in range(len(data_a)):
    xi = []
    for j in range(N-1, -1, -1):
        xi.append(data_a[i]**j)
    A.append(xi)
A = np.array(A)
B = np.array(data_b)

# LSE
x = LSE(A, B, lambd)
print("LSE:")
show_result(x)

# steepest descent
x_sd_l1 = steepest_descent(A, B, lambd)
print("Steepest Descent with L1 regularization:")
show_result(x_sd_l1)

# newton's method
x_newton = newton_method(A, B, lambd)
print("Newton's Method:")   
show_result(x_newton)
