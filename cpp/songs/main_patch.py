#! /usr/bin/env python

import os
import sys
import numpy as np
DEAR_PATH = '/mfs/user/panmiaocai/projects/mir_common'
sys.path.append(DEAR_PATH)
import dpark
from dear import get_song_path
from sid_label import get_song_ids_with_label_train, get_song_ids_with_label_test
from spec import process_mp3
import random

SIZE = 30
DIM = 513
SET = 'test'
DST_PATH = '/mfs/user/zhaojunbo/paracel/alg/ae/songs/dataset/patch/data_spec_' + SET


def map_list_to_str(dat):
    # normally, dat is a 2-D list or 2-D np.mat
    str_dat = ''
    for d_elem in dat:
        tmp =  map(lambda x : str(x) + ' ', d_elem)
        tmp += '\n'
        str_dat += reduce(lambda x, y: x + y, tmp)
    return str_dat


def map_song_with_label_te(data):
    """For testing cases, provide beansdb for python test script"""
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        new_mat = [new_mat[i] for i in wd_ids]
        assert all(len(mat_elem) == DIM for mat_elem in new_mat)
        assert len(new_mat) == SIZE
        _new_mat = np.array(new_mat)
        _new_mat = np.log(_new_mat + 1)
        _new_mat = normalize(_new_mat)
        new_mat = _new_mat.tolist() # IMPORTANT. beansdb can't save numpy.float64
        return label, new_mat
    except:
        import traceback
        traceback.print_exc()

    
def map_song_with_label_tr(data):
    """For paracel IO, provide textfiles"""
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        new_mat = [new_mat[i] for i in wd_ids]
        assert all(len(mat_elem) == DIM for mat_elem in new_mat)
        assert len(new_mat) == SIZE
        _new_mat = np.array(new_mat)
        _new_mat = np.log(_new_mat + 1)
        _new_mat = normalize(_new_mat)
        new_mat = map(lambda x: list(x), _new_mat)
        map(lambda x: x.append(label), new_mat)
        str_mat = map_list_to_str(new_mat)
        return str_mat
    except:
        import traceback
        traceback.print_exc()


def map_song_tr(data):
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        new_mat = [new_mat[i] for i in wd_ids]
        assert all(len(mat_elem) == DIM for mat_elem in new_mat)
        assert len(new_mat) == SIZE
        _new_mat = np.array(new_mat)
        _new_mat = np.log(_new_mat + 1)
        _new_mat = normalize(_new_mat)
        new_mat = map(lambda x: list(x), _new_mat)
        str_mat = map_list_to_str(new_mat)
        return str_mat
    except:
        import traceback
        traceback.print_exc()


def normalize(data):
    norm_data = data - np.mean(data, axis=1).reshape(data.shape[0], 1)
    # Truncate to +/-3 standard deviations and scale to -1 to 1
    dstd = 3 * np.std(norm_data)
    norm_data = np.maximum(np.minimum(norm_data, dstd), -dstd) / dstd
    # Rescale from [-1, 1] to [0.1, 0.9]
    norm_data = (norm_data + 1) * 0.4 + 0.1
    return norm_data


def main():
    dpark_ctx = dpark.DparkContext('mesos')

    # QA for overwrite case
    flag = False
    if len(os.listdir(DST_PATH)): # not empty
        while 1:
            print "Are you sure to overwrite the spectrum data in %s? [Y/n]" % DST_PATH
            key = raw_input()
            if key is 'Y':
                flag = True
                print 'Will overwrite immediately.'
                os.system('rm -rf ' + DST_PATH +  '/.[^.]') # TODO
                break
            elif key is 'n':
                flag = False
                print 'Will not overwrite it.'
                break
            else:
                print 'Please type [Y/n].'
    else:
        flag = True

    # Start extracting 
    if flag:
        if SET is 'train':
            # for paracel IO, provide Textfiles
            sids_with_label = get_song_ids_with_label_train()
            dpark_ctx.makeRDD(
                        sids_with_label, 100
                    ).map(
                        map_song_with_label_tr
                    ).filter(
                    lambda x: x != None
                    ).saveAsTextFile(
                        DST_PATH
                    )
        elif SET is 'test':
            # for python script of test, provide beansdb
            sids_with_label = get_song_ids_with_label_test(40000)
            dpark_ctx.makeRDD(
                        sids_with_label, 50
                    ).map(
                        map_song_with_label_te
                    ).filter(
                    lambda x: x != None
                    ).saveAsBeansdb(
                        DST_PATH
                    )
        else:
            print 'Not supported this data set'
            sys.exit(-1)

        print 'spec extracting done'


if __name__ == '__main__':
    main()
