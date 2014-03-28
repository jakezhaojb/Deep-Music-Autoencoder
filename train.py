# Sparse Autoencoder
# Multi-layer version.

# Instructions
# ------------

# This file is a framework for a stacked version of sparse autoencoder.
# It could be applied in some Machine Learning tasks. The information
# of interest is contained in the activation of the deepest layer of
# hidden units.

##====================================================================

import sys
sys.path.append("./visualization")
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sparse_autoencoder import *
from stack_and_para import vecstack2stack
from visualize import sample_image, display_effect
from dpark import DparkContext
from inspect import isfunction
import pickle
import pdb


def stocha_grad_desc_agagrad(fun_cost, fun_grad, theta, option, 
                             step_size_init=0.01, max_iter=10, tol=1e-7):
    mini_batch_size = 20

    assert(isfunction(fun_cost))
    assert(isfunction(fun_grad))
    assert(isinstance(theta, np.ndarray))

    for i in range(max_iter):
        
        mini_batch_ind = np.random.permutation(option[0].shape[1])

        for j in range(int(option[0].shape[1] / mini_batch_size)):
            cost = fun_cost(theta, *option)
            grad = fun_grad(theta, mini_batch_ind, j, *option)
            print "Big loop: %i, Small iter: %3i, Cost: %f" % (i, j, cost)
            
            # Adagrad
            try:
                adg
            except NameError:
                adg = grad ** 2
                step_size = step_size_init / np.sqrt(adg)
            else:
                adg = np.vstack((adg, grad ** 2))
                step_size = step_size_init / np.sqrt(np.sum(adg, axis=0))

            # momentum
            try:
                delta
            except NameError:
                delta = grad.copy()
            else:
                delta = delta * 0.5 + grad

            theta -= step_size * delta  # SGD with Adagrad and Momentum
                
            # Original SGD 
            #theta -= step_size_init * grad

            # Tolerance and stop iterating
            cost_pre = cost
            #if abs(cost_pre - cost) / max(1, cost, cost_pre) <= tol:
            #    print "The SGD has been converged under your tolerance."
            #    break

        del adg, delta

    return theta


def main():

    # Loading data
    print "Loading..."
    data_train = sample_image()

    # Initialize networks
    visible_size = 64  # number of input units
    hidden_size = [25, 16, 9]  # number of hidden units of each layer

    lamb = 0.0001     # weight decay parameter
    beta = 3    # weight of sparsity penalty dataset

    # dpark initialize
    dpark_ctx = DparkContext()

    # Start training, and L-BFGS is adopted
    # We apply a stack-wise greedy training process
    layer_ind = range(len(hidden_size) + 1)
    layer_ind.remove(0)
    layer_size = [visible_size] + hidden_size

    # desired average activation
    sparsity_param = dict()
    for ind in layer_ind:
        # standard: 64 units -> sparsity parameter 0.01
        sparsity_param[ind] = layer_size[ind - 1] * 0.01 / 64

    data = data_train
    opttheta = dict()  # parameter vector of stack AE
    img = dict()  # visualization mode

    for ind in layer_ind:

        print "start training layer No.%d" % ind

        # Obtain random parameters of considered layer
        theta = initial_parameter(layer_size[ind], layer_size[ind - 1])

        # SGD with mini-batch
        options = (data, layer_size[ind - 1], layer_size[ind],
                   beta, dpark_ctx)
        opttheta[ind] = stocha_grad_desc_agagrad(compute_cost, compute_grad,
                                                 theta, options)

        # Preparing next layer!
        W = opttheta.get(ind)[:layer_size[ind]*layer_size[ind-1]].\
            reshape(layer_size[ind], layer_size[ind-1])
        b = opttheta.get(ind)[2*layer_size[ind]*layer_size[ind-1]:\
            2*layer_size[ind]*layer_size[ind-1]+layer_size[ind]].\
            reshape(layer_size[ind], 1)
        data = sigmoid(np.dot(W, data) + b)

        # visulization shows
        img[ind] = display_effect(W)
        plt.axis('off')
        plt.savefig(str(ind) + '.jpg')

    # Trained parameters of stack AE
    para_stack = vecstack2stack(opttheta, hidden_size, visible_size)

    # Save trained weights and bias
    out = open("weights_bias.pkl", "wb")
    pickle.dump(para_stack, out)
    out.close()

    print "Mission complete!"


if __name__ == '__main__':
    main()
