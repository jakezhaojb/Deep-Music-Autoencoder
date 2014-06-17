#! /usr/bin/env python

# This file helps to finish a SOFTMAX framework using Vectorization.

import math
from random import randint
import numpy as np
import sys

N_ROW = 64
N_COL = 1000
CHECK_GRAD = True


def data_generator(n_row, n_col):
    # We write data by COL manner
    x = np.random.rand(n_row, n_col)
    y = []
    for i in range(n_col):
        y.append(randint(0, 4))
    return x, np.array(y)


def compute_prob(data_x, data_y, theta):
    # traverse the data
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (5, data_x.shape[0])
    prob = np.exp(np.dot(theta, data_x))
    norm_fact = np.sum(prob, axis=0)
    prob /= norm_fact
    return prob


def compute_cost(data_y, prob):
    assert prob.shape[1] == len(data_y)
    cost = 0
    lbl = np.zeros((len(data_y), 5))
    for lbl_elem, data_y_elem in zip(lbl, data_y):
        lbl_elem[data_y_elem] = 1
    cost_array = np.dot(lbl, np.log(prob))
    cost = cost_array.trace()
    cost *= -1. /  len(data_y)
    return cost
    

def compute_grad(data_x, data_y, prob):
    assert prob.shape[1] == len(data_y)
    lbl = np.zeros((data_x.shape[1], 5))
    for lbl_elem, data_y_elem in zip(lbl, data_y):
        lbl_elem[data_y_elem] = 1
    lbl = lbl.T - prob
    grad = np.dot(lbl, data_x.T)
    grad *= -1. / len(data_y)
    return grad


def main():
    rounds = 10000
    alpha = 0.001
    data_x, data_y = data_generator(N_ROW, N_COL)
    # initialize 5 thetas
    theta = np.random.rand(5, data_x.shape[0])

    if CHECK_GRAD:
        print 'Check gradients computing.'
        nume_grad = np.zeros(shape=theta.shape)
        for i, j in np.ndindex(theta.shape):
            theta_nume1 = theta.copy()
            theta_nume2 = theta.copy()
            theta_nume1[i, j] += 1e-4
            theta_nume2[i, j] -= 1e-4
            prob1 = compute_prob(data_x, data_y, theta_nume1)
            prob2 = compute_prob(data_x, data_y, theta_nume2)
            nume_grad[i, j] = compute_cost(data_y, prob1) -\
                              compute_cost(data_y, prob2)
            nume_grad[i, j] /= 2 * 1e-4
        grad =  compute_grad(data_x, data_y, \
                compute_prob(data_x, data_y, theta))
        print np.linalg.norm(grad - nume_grad) /\
                np.linalg.norm(grad + nume_grad)
        print 'Values printed above should be less than 1e-9'

    # Training
    while 1:
        print 'Start training [Y/n]?'
        key = raw_input()
        if key is 'Y':
            break
        elif key is 'n':
            sys.exit()
            print 'Quit here.'
        else:
            pass
    for i in range(rounds):
        # traverse the data
        prob = compute_prob(data_x, data_y, theta)
        print "cost is: %f" % compute_cost(data_y, prob)
        grad = compute_grad(data_x, data_y, prob)
        theta -= alpha * grad
            
if __name__ == '__main__':
    main()
