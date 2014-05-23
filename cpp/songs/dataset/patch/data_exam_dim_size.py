#! /usr/bin/env python
import os
import sys
sys.path.append('/mfs/user/zhaojunbo/paracel/alg/ae/songs')
from main_patch import SIZE, DIM
import time
import dpark

SET = 'train'
DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data_spec_' + SET))

    
def tr_data_len(fn):
    """This fun will examine the data's length, and rewrite it"""
    with open(fn) as fin:
        for i, l in enumerate(fin):
            try:
                assert len(l.split()) == DIM + 1, 'Data dimension not accorded'
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
        n_data = dpark_ctx.accumulator(0)
        def exam_each(data):
            assert len(data) == SIZE and all(len(dat) == DIM for dat in data)
            n_data.add(1)
        bool_rdd = dpark_ctx.beansdb(
                DATA_PATH
                ).map(
                    lambda (key, (val, _, __)):  val
                ).foreach(
                    exam_each
                )
        print "Total number: %i" % n_data.value
    

if __name__ == '__main__':
    main()
