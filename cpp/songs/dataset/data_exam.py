#! /usr/bin/env python
import os
import sys
sys.path.append('/mfs/user/zhaojunbo/paracel/alg/ae/songs')
from main import SIZE, DIM
import time
import dpark

SET = 'test'
DATA_PATH = '/mfs/user/zhaojunbo/paracel/alg/ae/songs/dataset/data_spec_' + SET

    
def tr_data_len(fn):
    """This fun will examine the data's length, and rewrite it"""
    with open(fn) as fin:
        for i, l in enumerate(fin):
            try:
                assert len(l.split()) == DIM * SIZE + 1, 'Data dimension not accorded'
            except:
                print 'error: %s, @line %i, Dim: %i' % (fn, i, len(l.split()))
                pass
    fin.close()
    print '%s, number: %i' % (fn, i + 1)
    return i + 1


def tr_data_len_rewrt(fn):
    """This fun will not only examine the data's length in the file, but also rewrite it"""
    fout = open(fn + '.tmp', 'w')
    j = 0
    with open(fn) as fin:
        for i, l in enumerate(fin):
            try:
                assert len(l.split()) == DIM * SIZE + 1, 'Data dimension not accorded'
                fout.write(l)
                j += 1
            except:
                #print 'error: %s, @line %i, Dim: %i' % (fn, i, len(l.split()))
                pass
    fin.close()
    fout.close()
    #os.remove(fn)
    os.rename(fn + '.tmp', fn)
    print 'Rewrite file: %s, number: %i' % (fn, j)
    return j


def main():
    dpark_ctx = dpark.DparkContext('mesos')
    if SET is 'train':
        print 'for training data:'
        assert os.path.isdir(DATA_PATH), 'Data dir not exist.'
        fn_list = filter(lambda x: not x.startswith('.'), os.listdir(DATA_PATH))
        fn_list = map(lambda x: os.path.join(DATA_PATH, x), fn_list)
        n_data = dpark_ctx.makeRDD(
                fn_list, 100
                ).map(
                tr_data_len        
                #tr_data_len_rewrt
                ).reduce(
                lambda x, y: x + y)
        print 'Total number : %i' % n_data

    if SET is 'test':
        print 'for testing data:'
        vec_rdd = dpark_ctx.beansdb(
                DATA_PATH
                ).map(
                    lambda (key, (val, _, __)):  val
                )
        data = vec_rdd.collect()
        assert all(len(dat[1]) == DIM * SIZE for dat in data), \
                'error of data dimension.'
        print "Total number: %i" % len(data)
    

if __name__ == '__main__':
    main()
