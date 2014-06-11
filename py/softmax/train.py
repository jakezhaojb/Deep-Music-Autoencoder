#! /usr/bin/env python

# This file will finish a SOFTMAX framework.
import math
from random import randint
import numpy as np

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


def compute_cost(data_y, prob):
    assert len(prob) == len(data_y)
    assert all(len(x.keys()) == 5 for x in prob)
    cost = 0
    for dat, prob_dat in zip(data_y, prob):
        prob_dat = dict(map(lambda (k, x): (k, int(dat == k) * math.log(x)),
                        prob_dat.items()))
        cost += reduce(lambda x, y: x + y, prob_dat.values())
    cost *= -1. / len(data_y)
    return cost


def compute_grad(data_x, data_y, prob):
    grad = {}
    for j in range(5):
        # for each theta
        grad_elem = np.zeros((data_x.shape[0],))
        for i, (dat_x, dat_y) in enumerate(zip(data_x.T, data_y)):
            # traverse dataset
            grad_elem += dat_x * (int(int(dat_y) == j) - prob[i].get(j))
        grad_elem *= -1. / len(data_y)
        grad[j] = grad_elem.reshape(grad_elem.size, 1)
    return grad


def compute_prob(data_x, data_y, theta):
    # traverse the data
    prob = []
    for dat in data_x.T:
        prob_elem_lbl = dict()
        for i in range(5):
            prob_elem_lbl[i] = np.exp(np.dot(theta[i].T,
                                      dat.reshape(dat.size, 1)))
        prob_sum = reduce(lambda x, y: x + y, prob_elem_lbl.values())
        for k in prob_elem_lbl.keys():
            prob_elem_lbl[k] = float(prob_elem_lbl.get(k) / prob_sum)
        prob.append(prob_elem_lbl)
    return prob


def main():
    rounds = 100
    alpha = 0.001
    data_x, data_y = data_generator(N_ROW, N_COL)
    # initialize 5 thetas
    theta = dict()
    for i in range(5):
        theta[i] = np.random.rand(data_x.shape[0], 1)

    if CHECK_GRAD:
        epsilon = np.eye(theta[0].shape[0], theta[0].shape[0]) * 1e-4
        for k in theta.keys():  # check it iterates every theta
            # check with theta[k]
            theta1 = theta.copy()
            theta2 = theta.copy()
            nume_grad = np.zeros(theta[k].shape)
            for i in range(theta.get(k).shape[0]):
                theta1[k] = theta.get(k) + epsilon[:, i].\
                                reshape(theta.get(k).shape[0], 1)
                theta2[k] = theta.get(k) - epsilon[:, i].\
                                reshape(theta.get(k).shape[0], 1)
                prob1 = compute_prob(data_x, data_y, theta1)
                prob2 = compute_prob(data_x, data_y, theta2)
                nume_grad[i] = compute_cost(data_y, prob1) -\
                               compute_cost(data_y, prob2)
                nume_grad[i] /= 2 * 1e-4
            prob = compute_prob(data_x, data_y, theta)
            grad = compute_grad(data_x, data_y, prob)
            print np.linalg.norm(grad.get(k) - nume_grad) /\
                    np.linalg.norm(grad.get(k) + nume_grad)
    print 'Values printed above should be less than 1e-9'

    # Training
    for i in range(rounds):
        # traverse the data
        prob = []
        for dat in data_x.T:
            prob_elem_lbl = dict()
            for i in range(5):
                prob_elem_lbl[i] = np.exp(np.dot(theta[i].T,
                                          dat.reshape(dat.size, 1)))
            prob_sum = reduce(lambda x, y: x + y, prob_elem_lbl.values())
            for k in prob_elem_lbl.keys():
                prob_elem_lbl[k] = float(prob_elem_lbl.get(k) / prob_sum)
            prob.append(prob_elem_lbl)

        cost = compute_cost(data_y, prob)
        grad = compute_grad(data_x, data_y, prob)
        assert grad.keys() == theta.keys()
        grad = dict(map(lambda x: (x[0], -alpha * x[1]), grad.items()))
        theta = dict(theta.items() + grad.items())
        print cost


if __name__ == '__main__':
    main()
