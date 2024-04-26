import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sum(x * np.log(x))

def gradient(x):
    return 1 + np.log(x)

def hessian(x):
    return np.diag(1 / x)

def newton_method(A, b, x0, max_iters, tolerance, beta, alpha):
    vals = [] #store values
    steps = [] #store newton steps
    x = x0.copy() #pass by ref

    for _ in range(max_iters):
        val = objective_function(x)
        vals.append(val)

        grad = gradient(x)
        hess = hessian(x)

        # solve system for the Newton direction v
        v = -np.linalg.solve(hess, grad)
        fprime = grad.dot(v)

        if abs(fprime) < tolerance: #check if threshold criteria met
            break

        # backtracking line search o.w.
        t = 1
        while objective_function(x + t * v) > val + alpha * t * fprime:
            t *= beta

        x += t * v
        steps.append(t)

    optval = vals[-1]

    #plot for obj fun error criteria
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(vals) - 1), np.array(vals[:-1]) - optval, '-', range(len(vals) - 1),
                 np.array(vals[:-1]) - optval, 'o')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Error')

# Problem setup
n = 100
m = 30
A = np.random.randn(m, n)
x_hat = np.random.rand(n) #uniform vector on [0,1]
b = A.dot(x_hat)

# Newton method parameters
max_iters = 1000
tolerance = 1e-6
beta = 0.5
alpha = 0.01

# initial point
x0 = x_hat.copy()

# run Newton's method
newton_method(A, b, x0, max_iters, tolerance, beta, alpha)
