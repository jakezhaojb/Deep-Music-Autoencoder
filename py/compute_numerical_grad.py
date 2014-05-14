# Instruction
# ------------
# This file is used to compute numerical gradients, which should be
# approximate to your computed gradients by Back-prepagation.

##================================================================
import numpy as np


def compute_numerical_grad(J, theta, *args):
    # Initialization
    numgrad = np.zeros(theta.shape[0])
    epsilon = 1e-4 * np.eye(theta.shape[0], theta.shape[0])
    for i in range(theta.shape[0]):
        numgrad[i] = J(theta + epsilon[i, :], *args) - \
            J(theta - epsilon[i, :], *args)
        numgrad[i] /= 2 * 1e-4
#        print numgrad[i]

    return numgrad


def check_compute_numgrad():
    x = np.array([4, 10])

    numgrad = compute_numerical_grad(simple_quadratic, x)

    grad = np.zeros((2, 1))
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]

    diff = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
    print diff


def simple_quadratic(x):
    value = x[0]**2 + 3*x[0]*x[1]
    return value
