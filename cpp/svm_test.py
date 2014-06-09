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
CROSS_VALIDATION = True


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


def data_dist(labels):
    GID_adj = range(len(GID))
    lbl_hist = []
    lbl_hist.extend([0] * len(GID_adj))
    for label_elem in labels:
        try:
            lbl_hist[GID_adj.index(label_elem)] += 1
        except:
            import traceback
            traceback.print_exc()
    for gid_elem, lbl_hist_elem in zip(GID_adj, lbl_hist):
        print '%i: %i' % (gid_elem, lbl_hist_elem)


def main():
    dpark_ctx = dpark.DparkContext('mesos')
    assert os.path.isdir(BASE_PATH) and os.path.isdir(MODEL_PATH)
    
    # Read the weights and bias of SDAE from MODEL_PATH
    print 'Loading the model'
    print 'Will adopt layer No. %i' % FEA_LAYER
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
    # TODO Involve the layer aiming at.
    lyr_last = lyr[FEA_LAYER]
    lyr = lyr[:FEA_LAYER]
    lyr.append(lyr_last)
    # SVM training data
    svm_data_tr_va = []
    svm_label_tr_va = []
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
    print 'Data aggregates now.'
    for i, data_agg_elem in enumerate(data_agg):
        assert len(data_agg_elem) == SIZE
        tmp_label = [_data_agg_elem[0] for _data_agg_elem in data_agg_elem]
        assert all(_tmp_label == tmp_label[0] for _tmp_label in tmp_label)
        tmp_data = np.array([_data_agg_elem[1] for _data_agg_elem in data_agg_elem])
        # TODO tmp_data.shape = (30, 30, 1), why?
        svm_data_tr_va.append(list(tmp_data.reshape(tmp_data.size,)))
        svm_label_tr_va.append(int(map_label(tmp_label[0])))
        if i % 1000 == 0:
            print 'Finish aggregate %i patch' % i
        svm_data_tr = svm_data_tr_va[10000:]
        svm_label_tr = svm_label_tr_va[10000:]
        svm_data_va = svm_data_tr_va[:10000]
        svm_label_va = svm_label_tr_va[:10000]
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
        svm_label_te.append(int(data_elem[0]))

    # Process data, view the GID distribution in tr or te sets
    print 'Processing data here.'
    print '=' * 100
    print 'Training data distributions:'
    data_dist(svm_label_tr)
    print '=' * 100
    print 'Testing data distributions:'
    data_dist(svm_label_te)
    print '=' * 100
    print 'Validation data distributions:'
    data_dist(svm_label_va)
    # Cross Validation
    if CROSS_VALIDATION:
        print 'SVM model starts cross validating.'
        del svm_data_tr, svm_data_te, svm_data_tr_va, \
                svm_label_tr, svm_label_te, svm_label_tr_va
        for svm_c in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
            svm_opt = '-c ' + str(svm_c) + ' '
            for gid_elem, va_hist_elem in zip(GID_adj, va_hist):
                wgt_tmp = max(va_hist) / float(va_hist_elem)
                '''
                if wgt_tmp < 3.0:
                    wgt = 1
                elif wgt_tmp < 10:
                    wgt = 4
                elif wgt_tmp < 40:
                    wgt = 16
                else:
                    wgt = 32
                '''
                if wgt_tmp < 10.0:
                    wgt = int(wgt_tmp)
                elif wgt_tmp < 40:
                    wgt = 16
                else:
                    wgt = 32
                svm_opt += ('-w' + str(gid_elem) + ' ' + str(wgt) + ' ')
            svm_opt += '-v 5 -q'
            print svm_opt
            svm_model = svm.svm_train(svm_label_va, svm_data_va, svm_opt)
        sys.exit(1)

    # SVM running
    fn_svm = 'svm_model_c0_01_wgt'
    if SAVE_OR_LOAD: # True
        print 'SVM model starts training.'
        svm_opt = '-c 0.01 '
        for gid_elem, tr_hist_elem in zip(GID_adj, tr_hist):
            wgt_tmp = max(tr_hist) / float(tr_hist_elem)
            if wgt_tmp < 3.0:
                wgt = 1
            elif wgt_tmp < 10:
                wgt = 2
            elif wgt_tmp < 40:
                wgt = 4
            else:
                wgt = 8
            svm_opt += ('-w' + str(gid_elem) + ' ' + str(wgt) + ' ')
        print svm_opt
        svm_model = svm.svm_train(svm_label_tr, svm_data_tr, svm_opt)
        # save SVM model
        svm.svm_save_model(fn_svm, svm_model)
    else: # False
        print 'SVM model loading.'
        # load SVM model
        svm_model = svm.svm_load_model(fn_svm)
    print 'SVM model training or loading done'
    p_label, p_acc, p_val = svm.svm_predict(svm_label_te, svm_data_te, svm_model)
    fid = open('res_tmp.pkl', 'wb')
    pickle.dump((p_label, p_acc, p_val), fid)
    fid.close()

if __name__ == '__main__':
    main()
