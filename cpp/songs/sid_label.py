# This file extracts the song_id of the songs which are involved with a "genre_ids"
# Also generate training and testing set, of song id and genra id tuples.
# ===============================
import os
import sys
import ast
import numpy as np

DATA_PATH = '/home2/dbsync/fb_song'
FILES = os.listdir(DATA_PATH)
GID = [331, 335, 325, 337, 328, 334, 336, 327, 326, 332, 333, 324, 329, 330]
GID.sort()

labels = []
sid = []


def get_song_ids_with_label():
    sids_with_label = np.load('dataset/song_ids_with_genre_ids.npy')
    return map(lambda x: tuple(x), sids_with_label)


def get_song_ids_with_label_train(size=None):
    sids_with_label_train = np.load('dataset/song_ids_with_genre_ids_train.npy')
    sids_with_label_train = sids_with_label_train[:size]
    return map(lambda x: tuple(x), sids_with_label_train)

def get_song_ids_with_label_test(size=None):
    sids_with_label_test = np.load('dataset/song_ids_with_genre_ids_test.npy')
    sids_with_label_test = sids_with_label_test[:size]
    return map(lambda x: tuple(x), sids_with_label_test)


def main(argv):
    # Generate whole set song ids with genre_ids
    print 'Generate whole set.'
    if os.path.isfile('dataset/song_ids_with_genre_ids.npy'):
        print "'dataset/song_ids_with_genre_ids.npy' was generated."
    else:
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
        np.save('dataset/song_ids_with_genre_ids.npy', np.array(zip(labels, sid)))
                    
    # Generate training and testing set, with averagly distributed GID
    if len(argv) is 1:
        flag = False; # whether generate?
        print 'To generate training and testing sets, usage: sid_label.py <training_set_size>'
    elif len(argv) is 2:
        if os.path.isfile('dataset/song_ids_with_genre_ids_train.npy'):
            while 1:
                print 'Are you sure to overwrite the training and testing sets file \
                        (song_ids_with_genre_ids_train.npy, song_ids_with_genre_ids_test.npy)? [Y/n]'
                key = raw_input()
                if key is 'Y':
                    flag = True
                    print 'Will regenerate training and testing sets.'
                    os.remove('dataset/song_ids_with_genre_ids_train.npy')
                    os.remove('dataset/song_ids_with_genre_ids_test.npy')
                    break
                elif key is 'n':
                    flag = False
                    print 'Will not overwrite training and testing sets file'
                    break
                else:
                    'Please type [Y/n]'
        else:
            flag = True
    else:
        print 'To generate training and testing sets, usage: sid_label.py <training_set_size>'
        sys.exit(-1)

    if flag:
        size = int(argv[1])
        print 'Generate training and testing sets.'
        sids_with_label = get_song_ids_with_label()
        labels = map(lambda x: x[0], sids_with_label) # Get the labels
        hist, bin_edges = np.histogram(labels, bins=GID)
        if size / len(GID) > hist.min():
            size = hist.min() * len(GID)
        print "Training set size: %i" % size
        print "Testing set size: %i" % (len(sids_with_label) - size)
        n_bin = size / len(GID) # Get the number of each GID
        # averagely distribute labels
        labels = np.array(labels)
        sids_with_label_train = []
        sids_with_label_test = []
        for gid in GID:
            idx_tr = list(np.where(labels == gid)[0])
            idx_tr = idx_tr[:n_bin]
            idx_te = list(set(range(len(labels))) - set(idx_tr))
            sids_with_label_gid_tr = [sids_with_label[i] for i in idx_tr]
            sids_with_label_gid_te = [sids_with_label[i] for i in idx_te]
            sids_with_label_train.extend(sids_with_label_gid_tr)
            sids_with_label_test.extend(sids_with_label_gid_te)
        np.save('dataset/song_ids_with_genre_ids_train.npy', np.array(sids_with_label_train))
        np.save('dataset/song_ids_with_genre_ids_test.npy', np.array(sids_with_label_test))

    print 'Generating done'


if __name__ == '__main__':
    main(sys.argv)
    
