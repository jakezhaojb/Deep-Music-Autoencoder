# coding: utf-8

import os
import sys
DEAR_PATH = '/mfs/user/panmiaocai/projects/mir_common'
sys.path.append(DEAR_PATH)
import dpark
from dear import get_song_path
from sid_label import get_song_ids_with_label, get_song_ids_train_set
from spec import process_mp3
import random

SIZE = 100

def map_song_with_label(data):
    label, sid = data
    try:
        song_path = get_song_path(sid)
        new_mat = process_mp3(song_path)
        wd_ids = range(len(new_mat))
        random.shuffle(wd_ids)
        wd_ids = wd_ids[:SIZE]
        _new_mat = [new_mat[i] for i in wd_ids]
        return sid, (label, _new_mat)
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
        _new_mat = [new_mat[i] for i in wd_ids]
        return _new_mat
    except:
        import traceback
        traceback.print_exc()



def main():
    dpark_ctx = dpark.DparkContext('mesos')
    sids_with_label = get_song_ids_with_label(100)
    # if with labels and sond_ids, FOR TESTING SET!
    # TODO
    dpark_ctx.makeRDD(
        sids_with_label, 50
    ).map(
        map_song_with_label
    ).saveAsBeansdb(
        'data_with_label'
    )

    # if without labels and song_ids, FOR TRAINING AE
    # TODO


if __name__ == '__main__':
    main()
