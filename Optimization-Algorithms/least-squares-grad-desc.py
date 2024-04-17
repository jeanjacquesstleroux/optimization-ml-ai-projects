#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:47:09 2024

@author: stleroux
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def initialize_problem(n, p):
    """ Initialize random problem dimensions and matrix A, and vector y. """
    np.random.seed(1)
    A = np.random.randn(n, p)  # n by p matrix
    y = np.random.randn(n)     # target vector
    x = np.zeros(p)            # initial x in R^p
    return A, y, x

def objective_function(A, x, y, n):
    """ Calculate the least squares objective function value. """
    return 1/(2 * n) * np.linalg.norm(y - A.dot(x))**2

def calculate_gradient(A, x, y, n):
    """ Calculate the gradient of the least squares objective. """
    return (1 / n) * np.dot(A.T, np.dot(A, x) - y)

def line_search(A, x, v, y, n, alpha, beta, current_val, fprime):
    """ Perform backtracking line search to find the appropriate step size. """
    t = 1.0
    while objective_function(A, x + t * v, y, n) > current_val + alpha * t * fprime:
        t *= beta
    return t

def main():
    # initialize constants
    n, p = 50, 6
    alpha, beta = 0.01, 0.5
    max_iterations, eta = 1000, 1e-3

    A, y, x = initialize_problem(n, p)

    # Gradient Descent
    vals, steps = [], []

    for _ in range(max_iterations):
        val = objective_function(A, x, y, n)
        vals.append(val)

        grad = calculate_gradient(A, x, y, n)
        v = -grad
        fprime = grad.dot(v)

        if np.linalg.norm(grad) < eta:
            print("Gradient norm below threshold, stopping.")
            break

        t = line_search(A, x, v, y, n, alpha, beta, val, fprime)
        x += t * v
        steps.append(t)

    # CVXPY solution
    x_cvx = cp.Variable(p)
    objective = cp.Minimize(cp.norm(A @ x_cvx - y)**2)
    prob = cp.Problem(objective)
    prob.solve()
    x_opt = x_cvx.value
    opt_val = objective_function(A, x_opt, y, n)
    
    print("Solution via Gradient Descent:", x)
    print("Solution via CVXPY:", x_opt)
    print("Optimal value (CVXPY):", opt_val)
    print("Status (CVXPY):", prob.status)

    # Plotting results
    plot_results(vals, opt_val)

def plot_results(vals, opt_val):
    plt.figure()
    errors = [np.log10(val - opt_val) for val in vals if val - opt_val > 0]
    plt.plot(errors, 'b-', label='log10(f(x^(k)) - f(x_opt))')
    plt.title('Logarithmic Error Reduction Over Iterations')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Log10 Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
