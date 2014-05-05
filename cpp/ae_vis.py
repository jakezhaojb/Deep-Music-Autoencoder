# This file can be adopted to visulize the weights of a trained AE.

import os
import sys
sys.path.append("/mfs/user/zhaojunbo/dpark/python_stack_AE_dpark_ReLU/visualization")
import numpy as np
import matplotlib.pyplot as plt
from visualize import display_effect


# Read Weights and Bias from .txt file
def read_wegiht_bias(fname):
    assert os.path.isfile(fname), "The WgtBias file not existed"
    wgt = np.loadtxt(fname)
    return wgt


def to_jpg(fname):
    if fname.endswith(".jpg"):
        return fname
    else:
        return fname + ".jpg"


def main(argv):
    assert len(argv) == 3, "Wrong command line parameters"
    wgt = read_wegiht_bias(argv[1])
    im = display_effect(wgt)
    plt.axis("off")
    plt.savefig(to_jpg(argv[2]))


if __name__ == '__main__':
    main(sys.argv)
