#! /usr/bin/env python

import os
import sys
sys.path.append('/mfs/user/zhaojunbo/libsvm-3.17/python')
sys.path.append('/mfs/user/zhaojunbo/paracel/alg/ae/songs')
from sid_label import GID
from main_patch import SIZE, DIM
import numpy as np
import dpark
import svmutil as svm
import pickle

BASE_PATH = '/mfs/user/zhaojunbo/paracel/alg/ae/songs/dataset/patch'
TRAIN_DATA_PATH = os.path.join(BASE_PATH, 'data_spec_train')
TEST_DATA_PATH = os.path.join(BASE_PATH, 'data_spec_test')
MODEL_PATH = os.path.join(BASE_PATH, 'output_sdae')
FEA_LAYER = -1
SAVE_OR_LOAD = True


def sigmoid(x):
    sigm = 1. / (1 + np.exp(-x))
    return sigm


def map_label(label):
    return GID.index(int(label))


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
    dpark_ctx = dpark.DparkContext('mesos')
    assert os.path.isdir(BASE_PATH) and os.path.isdir(MODEL_PATH)
    
    # Read the weights and bias of SDAE from MODEL_PATH
    print 'loading the model'
    W = dict()
    b = dict()
    n_lyr = len(os.listdir(MODEL_PATH)) / 4 # number of layers
    for _lyr in range(n_lyr):
        fn_name_W = 'ae_layer_' + str(_lyr) + '_W1'
        fn_name_W = os.path.join(MODEL_PATH, fn_name_W)
        fn_name_b = 'ae_layer_' + str(_lyr) + '_b1'
        fn_name_b = os.path.join(MODEL_PATH, fn_name_b)
        W[_lyr] = load_text_file(fn_name_W)
        b[_lyr] = load_text_file(fn_name_b)
    print 'model loading done'
    
    # SVM
    lyr = W.keys()
    lyr.sort()
    lyr = lyr[:FEA_LAYER]
    # SVM training data
    svm_data_tr = []
    svm_label_tr = []
    def map_once_tr(str_data):
        _str_data = str_data.split()
        key = float(_str_data[-1])
        val = map(lambda x: float(x), _str_data[:-1])
        assert len(val) == DIM, "SVM training data dim not accorded."
        val = np.array(val).reshape(DIM, 1)
        for lyr_elem in lyr:
            val = sigmoid(np.dot(W.get(lyr_elem), val) + b.get(lyr_elem))
        return key, val
    vec_rdd = dpark_ctx.textFile(
                TRAIN_DATA_PATH
            ).map(
                map_once_tr
            )
    data = vec_rdd.collect()
    data_agg = [data[x: x+SIZE] for x in xrange(0, len(data), SIZE)]
    assert data_agg[-1][-1] == data[-1] # QA
    print 'SVM training data starts aggregation.'
    for i, data_agg_elem in enumerate(data_agg):
        assert len(data_agg_elem) == SIZE
        tmp_label = [_data_agg_elem[0] for _data_agg_elem in data_agg_elem]
        assert all(_tmp_label == tmp_label[0] for _tmp_label in tmp_label)
        tmp_data = np.array([_data_agg_elem[1] for _data_agg_elem in data_agg_elem])
        # TODO tmp_data.shape = (30, 30, 1), why?
        svm_data_tr.append(list(tmp_data.reshape(tmp_data.size,)))
        svm_label_tr.append(map_label(tmp_label[0])) 
        if i % 1000 == 0:
            print 'Finish aggregate %i patch' % i
    # SVM testing data
    svm_label_te = []
    svm_data_te = []
    def map_once_te(data):
        key = map_label(int(data[0]))
        val = np.array(data[1][0]).T
        assert val.shape == (DIM, SIZE)
        for lyr_elem in lyr:
            val = sigmoid(np.dot(W.get(lyr_elem), val) + b.get(lyr_elem))
        val = reduce(lambda x, y: np.append(x, y), val) # TODO order
        val = val.reshape(val.size,).tolist() # TODO Code is a little too complicated
        return key, val
    vec_rdd = dpark_ctx.beansdb(
            TEST_DATA_PATH
            ).map(
                map_once_te
            )
    data = vec_rdd.collect()
    for data_elem in data:
        svm_data_te.append(data_elem[1])
        svm_label_te.append(data_elem[0])
    # SVM running
    print 'SVM model starts training.'
    svm_model = svm.svm_train(svm_label_tr, svm_data_tr, '-c 1')
    if SAVE_OR_LOAD: # True
        # saved SVM model
        fn_svm = 'svm_model_c1'
        svm.svm_save_model(fn_svm, svm_model)
    else: # False
        # load SVM model
        svm_model = svm.svm_load_model(fn_svm)
    print 'SVM model training done and saved'
    p_label, p_acc, p_val = svm.svm_predict(svm_label_te, svm_data_te, svm_model)

if __name__ == '__main__':
    main()
