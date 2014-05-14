# Instruction
# -----------
# This file mainly implements Back propagation, and adds weight decay and
# sparsity penalty, which are controlled by lamb and (beta, sparsity_param).
# This script implements AE by Rectified Linear Function for activation.

##==========================================================
import numpy as np
from stack_and_para import stack2vecstack
import pdb


def initial_parameter(hidden_size, visible_size):

    # Ng's initialization
    #r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    #W1 = np.random.rand(hidden_size, visible_size) * 2 * r - r
    #W2 = np.random.rand(visible_size, hidden_size) * 2 * r - r

    # Initialize weight matrix to a mean 0 uniform distribution
    W1 = np.random.uniform(-1, 1, [hidden_size, visible_size])
    W2 = np.random.uniform(-1, 1, [visible_size, hidden_size])

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    # Convert weights and bias to the vector form
    theta = np.hstack(([], W1.reshape(W1.size),
                      W2.reshape(W2.size), b1, b2))

    return theta

    '''
    # One-step initialize whole framework
    assert(isinstance(hidden_size, dict))
    layer_ind = range(len(hidden_size) + 1)
    layer_ind.remove(0)
    layer_size = [visible_size] + hidden_size

    W = dict()
    b = dict()

    # Initialization
    for ind in layer_ind:
        W[ind] = dict()
        b[ind] = dict()
        r = np.sqrt(6) / np.sqrt(layer_size[ind] + layer_ind[ind-1] + 1)
        W[ind][1] = np.random.rand(layer_size[ind], layer_size[ind-1]) *\
            2 * r - r
        W[ind][2] = np.random.rand(layer_size[ind-1], layer_size[ind]) *\
            2 * r - r
        b[ind][1] = np.zeros(layer_size[ind])
        b[ind][2] = np.zeros(layer_size[ind-1])

    stack = (W, b)

    # Convert to vector form
    theta = stack2vecstack(stack)

    return theta
    '''


def sigmoid(x):
    # Regularly, I choose sigmoid function as the active function.
    if not isinstance(x, np.ndarray):
        print "Wrong parameter of sigmoid function"
        return False
    sigm = 1.0 / (1 + np.exp(-x))
    return sigm


def ReLU(x):
    # Recifier Nonlinearities
    if not isinstance(x, np.ndarray):
        print "Wrong parameter of ReLU funciton"
        return False
    relu = x.copy()  # copy(), Too Important!
    relu[relu < 0] = 0
    return relu


def compute_cost(theta, *args):

    # SGD with mini-batch parameters
    assert(len(args) > 2 and len(args) < 6)

    lamb = 0.0001
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]

    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        dpark = args[4]

    # Initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers

    # Get parameters from theta of vector version
    W1 = theta[: hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size: 2 * hidden_size * visible_size].\
        reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size +
               hidden_size].reshape(hidden_size, 1)
    b2 = theta[2 * hidden_size * visible_size + hidden_size:].\
        reshape(visible_size, 1)

    # dpark version
    cost_acc = dpark.accumulator(0)
    sparsity_stat_acc = dpark.accumulator(0)

    # Broadcast
    W1 = dpark.broadcast(W1)
    W2 = dpark.broadcast(W2)
    b1 = dpark.broadcast(b1)
    b2 = dpark.broadcast(b2)
    def map_iter(dat):
        a[1] = dat.reshape(visible_size, 1)
        z[2] = np.dot(W1, a[1]) + b1
        a[2] = ReLU(z[2])
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = ReLU(z[3])

        # To see whether sparsity increases
        sparsity_stat_acc.add(len(np.where(a[2] == 0)[0]) / float(len(a[2])))

        cost_acc.add(np.sum(np.power(a[3] - a[1], 2)) / 2)

    #print "compute_cost rho collecting"
    dpark.makeRDD(
        data.T, 200
        ).map(
        map_iter
        ).collect()
    cost = cost_acc.value
    print "mean sparsity: %f" % (sparsity_stat_acc.value / float(data.shape[1]))

    # Broadcast
    W1.clear()
    W2.clear()
    b1.clear()
    b2.clear()
    # dpark finished

    # No! We will adopt SGD
    cost = cost / data.shape[1]
    cost += lamb / 2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))  # better than the former
    return cost


def compute_grad(theta, mini_batch_ind, mini_batch_size, index_loop, *args):

    # SGD with mini-batch parameters
    assert(len(args) > 2 and len(args) < 6)

    lamb = 0.0001
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]

    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        dpark = args[4]

    # Get parameters from theta of vector version
    W1 = theta[: hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size: 2 * hidden_size * visible_size].\
        reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size +
               hidden_size].reshape(hidden_size, 1)
    b2 = theta[2 * hidden_size * visible_size + hidden_size:].\
        reshape(visible_size, 1)

    # initialize gradients
    W1_grad = np.zeros(W1.shape)
    W2_grad = np.zeros(W2.shape)
    b1_grad = np.zeros(b1.shape)
    b2_grad = np.zeros(b2.shape)

    # initialize delta items
    W1_delta = np.zeros(W1.shape)
    W2_delta = np.zeros(W2.shape)
    b1_delta = np.zeros(b1.shape)
    b2_delta = np.zeros(b2.shape)

    # initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers
    sigma = dict()  # keys are from 2 to number of layers
    a_der = dict()  # ReLU as activation function

    # dpark version
    # new version
    # Backpropogation

    # Broadcast
    W1 = dpark.broadcast(W1)
    W2 = dpark.broadcast(W2)
    b1 = dpark.broadcast(b1)
    b2 = dpark.broadcast(b2)

    def map_der_iter(dat):
        a[1] = dat.reshape(visible_size, 1)
        z[2] = np.dot(W1, a[1]) + b1
        a[2] = ReLU(z[2])
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = ReLU(z[3])

        # Deriverates of ReLU function
        a_der[3] = a[3].copy()  # copy(), Too Important!
        a_der[3][np.where(a_der[3] > 0)] = 1
        a_der[2] = a[2].copy()  # copy(), Too Important!
        a_der[2][np.where(a_der[2] > 0)] = 1

        # sigma computing
        sigma[3] = -(a[1] - a[3]) * a_der[3] 
        sigma[2] = (np.dot(W2.T, sigma[3])) * a_der[2]

        res = (np.dot(sigma[3], a[2].T),
               np.dot(sigma[2], a[1].T),
               sigma[3], sigma[2])

        return res

    para_collect = dpark.makeRDD(
                    data.T[
                    mini_batch_ind[
                    index_loop*mini_batch_size:\
                    (index_loop+1)*mini_batch_size
                    ], :]).map(
                    map_der_iter
                    ).reduce(
                    lambda x, y: (
                    x[0]+y[0], x[1]+y[1], 
                    x[2]+y[2], x[3]+y[3]
                    ))
    
    W2_delta = para_collect[0]
    W1_delta = para_collect[1]
    b2_delta = para_collect[2]
    b1_delta = para_collect[3]

    # gradient computing
    W1_grad = W1_delta / mini_batch_size + lamb * W1
    W2_grad = W2_delta / mini_batch_size + lamb * W2
    b1_grad = b1_delta / mini_batch_size
    b2_grad = b2_delta / mini_batch_size

    # Broadcast
    W1.clear()
    W2.clear()
    b1.clear()
    b2.clear()
    # dpark finished

    # return vector version 'grad'
    grad = np.hstack(([], W1_grad.reshape(W1.size), W2_grad.reshape(W2.size),
                      b1_grad.reshape(b1.size), b2_grad.reshape(b2.size)))

    return grad
