import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def initialize_problem(m, n):
    """initialize random problem dimensions and matrix A"""
    np.random.seed(1)
    A = np.random.randn(m, n)  # m by n matrix
    x = np.zeros(n)  # x in R^n
    return A, x

def objective_function(A, x):
    """calculate the objective function value"""
    Ax = A.dot(x)
    return -np.sum(np.log(1 - Ax)) - np.sum(np.log(1 + x)) - np.sum(np.log(1 - x))

def calculate_gradient(A, x):
    """calculate the gradient of the objective function"""
    Ax = A.dot(x)
    d = 1 / (1 - Ax)
    return A.T.dot(d) - 1 / (1 + x) + 1 / (1 - x)

def line_search(A, x, v, alpha, beta, current_val, fprime):
    """perform backtracking line search to find the appropriate step size"""
    t = 1.0
    while np.max(A.dot(x + t * v)) >= 1 or np.max(np.abs(x + t * v)) >= 1:
        t *= beta
    while -np.sum(np.log(1 - A.dot(x + t * v))) - np.sum(np.log(1 - (x + t * v)**2)) > current_val + alpha * t * fprime:
        t *= beta
    return t

def solve_cvxpy(A, m, n):
    """solve the problem using CVXPY"""
    x = cp.Variable(n)
    objective = cp.Minimize(-cp.sum(cp.log(1 - A @ x)) - cp.sum(cp.log(1 + x)) - cp.sum(cp.log(1 - x)))
    constraints = [A @ x <= 1 - 1e-5, x <= 1 - 1e-5, x >= -1 + 1e-5]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value, problem.value, problem.status

def main():
    m, n = 200, 100
    A, x = initialize_problem(m, n)
    
    # Solve using manual gradient descent
    alpha, beta = 0.01, 0.5
    max_iterations, eta = 1000, 1e-3
    vals, steps = [], []
    for _ in range(max_iterations):
        val = objective_function(A, x)
        vals.append(val)
        
        grad = calculate_gradient(A, x)
        v = -grad
        fprime = grad.dot(v)
        
        if np.linalg.norm(grad) < eta:
            break
        
        t = line_search(A, x, v, alpha, beta, val, fprime)
        x += t * v
        steps.append(t)

    # Solve using CVXPY
    x_opt, opt_val, status = solve_cvxpy(A, m, n)

    # Plotting results
    plt.figure(1)
    plt.semilogy(range(len(vals)), vals, label='Objective Value (Manual)')
    plt.title('Objective Function Value Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()

    plt.figure(2)
    plt.plot(steps, label='Step Sizes')
    plt.title('Step Sizes Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Step Size')
    plt.legend()
    
    plt.show()

    print("Optimal x (CVXPY):", x_opt)
    print("Optimal value (CVXPY):", opt_val)
    print("Status (CVXPY):", status)

if __name__ == "__main__":
    main()
