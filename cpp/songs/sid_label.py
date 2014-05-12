# This file extracts the song_id of the songs which are involved with a "genre_ids"
# ===============================
import os
import sys
import ast
import numpy as np

DATA_PATH = '/home2/dbsync/fb_song'
FILES = os.listdir(DATA_PATH)
GID = [331, 335, 325, 337, 328, 334, 336, 327, 326, 332, 333, 324, 329, 330]

labels = []
sid = []


def get_song_ids_with_label(size=4000):
    sids_with_label = np.load('song_ids_with_genre_ids.npy')
    sids_with_label = sids_with_label[:size]
    return map(lambda x: tuple(x), sids_with_label)


def get_song_ids_train_set(sids_with_label):
    # TODO


def main():
    # extracting song ids of which involve genre_ids
    for filename in FILES:
        filename = os.path.join(DATA_PATH, filename)
        print "process the file: " + filename
        if not filename.endswith('.csv'):
            continue
        fin = open(filename)
        while 1:
            line = fin.readline()
            if not len(line):
                break
            try:
                info_dict = ast.literal_eval(line[line.find('{') : line.find('}') + 1])
                if not info_dict.get('genre_ids') is (None or ''):
                    gid = int(info_dict.get('genre_ids')) 
                    assert gid in GID
                    labels.append(gid)
                    sid.append( int(line[:line.find('\t')]) )
            except:
                # error occurs
                pass
        fin.close()
    # save the .npy file 
    np.save('song_ids_with_genre_ids.npy', np.array(zip(labels, sid)))
                

if __name__ == '__main__':
    main()
    
