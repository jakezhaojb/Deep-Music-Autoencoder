# Instruction
# -----------
# This file is used to help visualize the effect of Stack_AE
# It involves the function used to sample images, and visualize
# what kind of images the hidden layer is chasing for.

##============================================================

import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Return 10000 patches for training
def sample_image():
    assert(os.path.isfile("./visualization/IMAGES.mat"))

    IMAGES = sio.loadmat('./visualization/IMAGES.mat')['IMAGES']
    patchsize = 8
    numpatches = 10000

    # Initialize patches with zeros
    patches = np.zeros((patchsize*patchsize, numpatches))

    x = np.random.permutation(64)
    y = np.random.permutation(64)
    x = x[:40]
    y = y[:40]
    for i in range(10):
        img = IMAGES[:, :, i]
        for j in range(25):
            for k in range(40):
                pos_x = patchsize * x[j]
                pos_y = patchsize * y[k]
                patch = img[pos_x: pos_x + patchsize, pos_y: pos_y + patchsize]
                patch = patch.reshape(patchsize**2, 1, order='F')
                patches[:, 1000*i + 40*j + k] = patch.squeeze()

    # Since this framework adopts ReLU function, normalization is not in need.
    #patches = normalize_data(patches)


    return patches


# Squash data to [0.1, 0.9] since sigmoid function is applied.
def normalize_data(data):
    # Remove DC
    norm_data = data - np.mean(data, axis=0)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    dstd = 3 * np.std(norm_data)
    norm_data = np.maximum(np.minimum(norm_data, dstd), -dstd) / dstd

    # Rescale from [-1, 1] to [0.1, 0.9]
    norm_data = (norm_data + 1) * 0.4 + 0.1

    return norm_data


'''
# visualize the effect of networks
def display_effect(W):
    img_num, unit_num = W.shape
    assert(issquare(img_num) and issquare(unit_num))  # QA

    # Find out the input pixels that maximally activate hidden units
    x = np.zeros(shape=(img_num, unit_num))
    for i in range(img_num):
        # See 'VIsualization' chapter
        max_act = W[i, :] / np.linalg.norm(W[i, :])
        x[i, :] = max_act.squeeze()

    # Visualize the effect in a big image grid, in which each small block
    # matches the feature each considered hidden unit looks for.
    num = int(np.sqrt(img_num))
    unit_shape = int(np.sqrt(unit_num))
    visualgrid = np.zeros(shape=(num * (unit_shape + 2) - 2,
                                 num * (unit_shape + 2) - 2))
    # Constructing the image grids
    for i in range(num):
        for j in range(num):
            unit_x = x[i * num + j, :].reshape(unit_shape, unit_shape)
            x_ori = i * (unit_shape + 2)
            y_ori = j * (unit_shape + 2)
            visualgrid[x_ori: x_ori + unit_shape,
                       y_ori: y_ori + unit_shape] = unit_x

    im = plt.imshow(visualgrid, cmap=cm.gray, vmax=1, vmin=0)
    return im
'''


# visualize the effect of networks
def display_effect(W):
    img_num, pixel_num = W.shape
    assert issquare(pixel_num)  # QA

    # Find out the input pixels that maximally activate hidden units
    x = np.zeros(shape=(img_num, pixel_num))
    for i in range(img_num):
        # See 'VIsualization' chapter
        max_act = W[i, :] / np.linalg.norm(W[i, :])
        x[i, :] = max_act.squeeze()

    # Adjust img_num
    max_img_num = 100
    img_num = img_num if img_num < max_img_num else max_img_num
    perfect_square = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    if not issquare(img_num):
        perfect_square.append(img_num)
        perfect_square.sort()
        img_num = perfect_square[perfect_square.index(img_num) - 1]

    # Adjust X with adjusted img_num
    adj_x_ind = np.random.permutation(x.shape[0])
    adj_x_ind = adj_x_ind[:img_num]
    x = x[adj_x_ind, :]

    # Visualize the effect in a big image grid, in which each small block
    # matches the feature each considered hidden unit looks for.
    num = int(np.sqrt(img_num))
    unit_shape = int(np.sqrt(pixel_num))
    visualgrid = np.zeros(shape=(num * (unit_shape + 2) - 2,
                                 num * (unit_shape + 2) - 2))
    # Constructing the image grids
    for i in range(num):
        for j in range(num):
            unit_x = x[i * num + j, :].reshape(unit_shape, unit_shape)
            x_ori = i * (unit_shape + 2)
            y_ori = j * (unit_shape + 2)
            visualgrid[x_ori: x_ori + unit_shape,
                       y_ori: y_ori + unit_shape] = unit_x

    im = plt.imshow(visualgrid, cmap=cm.gray, vmax=1, vmin=0)
    return im


# Still a tiny problem here
def issquare(integer):
    root = math.sqrt(integer)
    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False
