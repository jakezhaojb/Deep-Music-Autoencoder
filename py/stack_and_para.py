# Instruction
# -----------
# This file can help you convert the weights or bias stack,
# as "dict" in python, to a parameter vector.
# Accordingly, the conversion from vector to stack is as well
# implemented.

# The stacks are constructed as a tuple, combined by weight dict
# & bias dict. Both dicts are formed layer by layer, whose keys
# indicate layer indexes and items are weights or bias of the
# considered layer.
# Usage:
#   W = stack[0]
#   b = stack[1]

# Note that this framework is comprised of multiple stacks AE:
# W[1][1], W[1][2], b[1][1], b[1][2] are the parameters of the
# first AE obtained by Back-propagation.

# Remember, the hidden layer number is 1 less than the weight &
# bias number!
##=============================================================

import numpy as np


def stack2vecstack(stack):
    """Convert parameter stack to a vector"""
    assert(isinstance(stack, tuple))
    assert(stack[0].keys() == stack[1].keys())
    layer_num = max(stack[0].keys())
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)
    assert(stack[0].keys() == layer_ind)

    # Conversion
    theta = dict()
    W = stack[0]
    b = stack[1]
    for k in layer_ind:
        assert(len(W[k].keys()) == 2)
        assert(len(b[k].keys()) == 2)
        assert(W[k][1].shape == W[k][2].shape[::-1])
        assert(W[k][1].shape[0] == b[k][1].shape[0])
        assert(W[k][2].shape[0] == b[k][2].shape[0])

        theta[k] = np.hstack((W[k][1].reshape(W[k][1].size),
                              W[k][2].reshape(W[k][2].size),
                              b[k][1].reshape(b[k][1].size),
                              b[k][2].reshape(b[k][2].size)))
    return theta


def vecstack2stack(theta, hidden_size, visible_size):
    assert(isinstance(theta, dict))
    layer_ind = range(len(hidden_size) + 1)
    layer_ind.remove(0)
    assert(theta.keys() == layer_ind)

    # Whole network size
    layer_size = [visible_size] + hidden_size

    # weight parsing
    weight = dict()
    bias = dict()
    for ind in layer_ind:
        weight[ind] = dict()
        bias[ind] = dict()
        pre_pos = 0
        pos = pre_pos + layer_size[ind]*layer_size[ind-1]
        weight[ind][1] = theta.get(ind)[pre_pos: pos].\
            reshape(layer_size[ind], layer_size[ind-1])
        pre_pos = pos
        pos += layer_size[ind-1] * layer_size[ind]
        weight[ind][2] = theta.get(ind)[pre_pos: pos].\
            reshape(layer_size[ind-1], layer_size[ind])
        pre_pos = pos
        pos += layer_size[ind]
        bias[ind][1] = theta.get(ind)[pre_pos: pos].\
            reshape(layer_size[ind], 1)
        pre_pos = pos
        pos += layer_size[ind-1]
        bias[ind][2] = theta.get(ind)[pre_pos: pos].\
            reshape(layer_size[ind-1], 1)

        '''
        pos = 0
        weight[ind][1] = theta.get(ind)
                [pos: pos+layer_size[ind]*layer_size[ind-1]].\
                reshape(layer_size[ind], layer_size[ind-1])
        pos += layer_size[ind-1] * layer_size[ind]
        weight[ind][2] = theta.get(ind)
                [pos: pos+layer_size[ind-1]*layer_size[ind]].\
                reshape(layer_size[ind-1], layer_size[ind])
        pos += layer_size[ind] * layer_size[ind-1]
        bias[ind][1] = theta.get(ind)[pos: pos+layer_size[ind]].\
            reshape(layer_size[ind], 1)
        pos += layer_size[ind]
        bias[ind][2] = theta.get(ind)[pos: pos+layer_size[ind-1]].\
            reshape(layer_size[ind-1], 1)
        pos += layer_size[ind-1]
        '''
        assert(len(theta.get(ind)) == pos)  # QA

    # stack forming, as a tuple
    stack = (weight, bias)

    return stack
