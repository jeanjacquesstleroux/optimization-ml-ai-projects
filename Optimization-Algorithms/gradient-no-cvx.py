#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:27:55 2024

@author: stleroux
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(1500)  # need to use so code can do the recursive calls

def initialize_problem(m, n):
    """initialize random problem dimensions and matrix a"""
    np.random.seed(1)
    A = np.random.randn(m, n) # m by n matrix
    x = np.zeros(n) # x in Rn
    return A, x

def objective_function(A, x):
    """calculate the objective function value"""
    Ax = A.dot(x)
    return -np.sum(np.log(1 - Ax)) - np.sum(np.log(1 + x)) - np.sum(
        np.log(1 - x))

def calculate_gradient(A, x):
    """calculate the gradient of the objective function"""
    Ax = A.dot(x)
    d = 1 / (1 - Ax)
    return A.T.dot(d) - 1 / (1 + x) + 1 / (1 - x)

def line_search(A, x, v, alpha, beta, current_val, fprime):
    """perform backtracking line search to find the appropriate step size"""
    t = 1.0
    # check for feasibility in the constraints
    while np.max(A.dot(x + t * v)) >= 1 or np.max(np.abs(x + t * v)) >= 1:
        t *= beta
    # check for sufficient decrease condition
    while -np.sum(np.log(1 - A.dot(x + t * v))) - np.sum(
            np.log(1 - (x + t * v)**2)) > current_val + alpha * t * fprime:
        t *= beta
    return t

def main():
    # constants
    m, n = 200, 100 # n and m as instructed in problem
    alpha, beta = 0.01, 0.5 #alpha/beta as recommended in problem
    max_iterations, eta = 1000, 1e-3 #stopping criteria, eta as suggested

    # initialize problem
    A, x = initialize_problem(m, n)
    
    # optimization variables
    vals, steps = [], []

    # main optimization loop
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

    # plotting results
    plot_results(vals, steps)

def plot_results(vals, steps):
    """plot the results"""
    # obj function value over iterations plot
    plt.figure(1)
    plt.semilogy(range(len(vals) - 1), np.array(vals[:-1]) - vals[-1], '-') # log scale axis
    plt.title('Objective Function Value Over Iterations') 
    plt.xlabel('Iteration (k)')
    plt.ylabel('Function Values (' + r'$f(\mathbf{x}^{(k)}) - p^*$' + ')')

    # step sizes over iterations plot
    plt.figure(2)
    plt.plot(range(len(steps)), steps, 'o:', label='Step Size')
    plt.title('Step Sizes Over Iterations')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Step Size (' + r'${t}^{(k)}$' + ')')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
