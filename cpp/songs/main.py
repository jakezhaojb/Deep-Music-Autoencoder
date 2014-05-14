# coding: utf-8

import os
import sys
import numpy as np
DEAR_PATH = '/mfs/user/panmiaocai/projects/mir_common'
sys.path.append(DEAR_PATH)
import dpark
from dear import get_song_path
from sid_label import get_song_ids_with_label_train
from spec import process_mp3
import random

SIZE = 30


def map_list_to_str(dat):
    # normally, dat is a 2-D list or 2-D np.mat
    str_dat = ''
    for d_elem in dat:
        tmp =  map(lambda x : str(x) + ' ', d_elem)
        str_dat += reduce(lambda x, y: x + y, tmp)
    return str_dat


def map_song_with_label(data):
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        new_mat = [new_mat[i] for i in wd_ids]
        new_mat = np.array(new_mat)
        new_mat = np.log(new_mat + 1)
        new_mat = normalize(new_mat)
        str_mat = map_list_to_str(new_mat)
        str_mat += str(label) # with label
        return str_mat
    except:
        import traceback
        traceback.print_exc()


def map_song(data):
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        new_mat = [new_mat[i] for i in wd_ids]
        new_mat = np.array(new_mat)
        new_mat = np.log(new_mat + 1)
        new_mat = normalize(new_mat)
        str_mat = map_list_to_str(new_mat)
        return str_mat
    except:
        import traceback
        traceback.print_exc()


def normalize(data):
    norm_data = data - np.mean(data, axis=0)
    # Truncate to +/-3 standard deviations and scale to -1 to 1
    dstd = 3 * np.std(norm_data)
    norm_data = np.maximum(np.minimum(norm_data, dstd), -dstd) / dstd
    # Rescale from [-1, 1] to [0.1, 0.9]
    norm_data = (norm_data + 1) * 0.4 + 0.1
    return norm_data


def main():
    dpark_ctx = dpark.DparkContext('mesos')
    sids_with_label = get_song_ids_with_label_train()
    dpark_ctx.makeRDD(
        sids_with_label, 50
    ).map(
        map_song_with_label
    ).saveAsTextFile(
        'dataset/data_spec'
    )
    print 'spec extracting done'


if __name__ == '__main__':
    main()
