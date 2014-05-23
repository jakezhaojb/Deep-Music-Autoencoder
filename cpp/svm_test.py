#! /usr/bin/env python

import os
import sys
sys.path.append('mfs/user/zhaojunbo/libsvm-3.17/python')
sys.path.append('mfs/user/zhaojunbo/paracel/alg/ae/songs')
from sid_label import GID
from main_patch import SIZE, DIM
import numpy as np
import dpark
from svmutil import svm

BASE_PATH = 'mfs/user/zhaojunbo/paracel/alg/ae/songs/dataset/patch'
TRAIN_DATA_PATH = os.path.join(BASE_PATH, 'data_spec_train')
TEST_DATA_PATH = os.path.join(BASE_PATH, 'data_spec_test')
MODEL_PATH = os.path.join(BASE_PATH, 'output_sdae')
FEA_LAYER = -1


def sigmoid(x):
    sigm = 1. / (1 + np.exp(-x))
    return sigm


def map_label(label):
    return GID.index(label)


def load_text_file(filename):
    fin = open(filename)
    res = []
    while 1:
        line = fin.readline()
        if len(line) == 0:
            break
        res.append(map(lambda x: float(x), line.split()))
    return np.array(res)


def main():
    dpark_ctx = dpark.DparkContext()
    assert os.path.isdir(DATA_PATH) and os.path.isdir(MODEL_PATH)
    
    # Read the weights and bias of SDAE from MODEL_PATH
    print 'loading the model'
    W = dict()
    b = dict()
    n_lyr = len(os.listdir(MODEL_PATH)) / 2 # number of layers
    for lyr in range(4):
        fn_name_W = 'ae_layer_' + str(lyr) + '_W1'
        fn_name_W = os.path.join(MODEL_PATH, fn_name_W)
        fn_name_b = 'ae_layer_' + str(lyr) + '_b1'
        fn_name_b = os.path.join(MODEL_PATH, fn_name_W)
        W[lyr] = load_text_file(fn_name_W)
        b[lyr] = load_text_file(fn_name_b)
    print 'model loading done'
    
    # SVM model
    lyr = W.keys()
    lyr.sort()
    lyr = lyr[:FEA_LAYER]
    # SVM training
    svm_data_tr = []
    svm_label_tr = []
    def map_once_tr(str_data):
        _str_data = str_data.split()
        key = float(_str_data[-1])
        val = map(lambda x: float(x), _str_data[:-1])
        assert len(val) == DIM, "val dim not accorded."
        for lyr_elem in lyr:
            val = sigmoid(np.dot(W.get(lyr_elem), val) + b.get(lyr_elem)) # TODO, whether tranform?
        return key, val
    vec_rdd = dpark_ctx.textFile(
                TRAIN_DATA_PATH
            ).map(
                map_once_tr
            )
    data = vec_rdd.collect()
    data_agg = [l[x: x+SIZE] for x in xrange(0, len(data), SIZE)]
    assert data_agg[-1][-1] == data[-1] # QA
    print 'SVM training data starts aggregation.'
    for data_agg_elem in data_agg:
        assert len(data_agg_elem) == SIZE
        tmp_label = [_data_agg_elem[0] for _data_agg_elem in data_agg_elem]
        tmp_data = np.array([_data_agg_elem[1] for _data_agg_elem in data_agg_elem])
        assert all(_tmp_label == tmp_label[0] for _tmp_label in tmp_label)
        svm_data_tr.append(list(tmp_data.reshape(DIM * SIZE,)))
        svm_label_tr.append(map_label(tmp_label[0])) 
    print 'SVM model starts training.'
    svm_model = svm.svm_train(svm_label_tr, svm_data_tr)
    print 'SVM model training done'
    # SVM testing
    svm_label_te = []
    svm_data_te = []
    def map_once_te(data):
        key = map_label(int(data[0]))
        val = np.array(data[1][0])
        for lyr_elem in lyr:
            val = sigmoid(np.dot(W.get(lyr_elem), val) + b.get(lyr_elem)) # TODO, whether tranform?
        val_fea = reduce(lambda x, y: np.append(x, y), val) # TODO order
        val_fea = list(val.reshape(DIM*SIZE,))
        return key, val_fea
    vec_rdd = dpark_ctx.beansdb(
            DATA_PATH
            ).map(
                map_once_te
            ).reduceByKey(
                lambda x, y: x + y
            )
    data = vec_rdd.collect()
    for data_elem in data:
        svm_data_te.extend(data_elem[1])
        svm_label_te.extend(data_elem[0] * len(data_elem))
    p_label, p_acc, p_val = svm.svm_predict(svm_label_te, svm_data_te, svm_model)

if __name__ == '__main__':
    main()
