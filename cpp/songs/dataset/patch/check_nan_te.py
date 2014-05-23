#! /usr/bin/env python

import os
import sys
import math
import dpark
import numpy as np


def main(argv):
    assert len(argv) == 2 or len(argv) == 3, "<usage> ./check_nan_te.py dirname rewrt_flag"
    assert os.path.isdir(argv[1])
    print 'Check nan in testing dir: %s' % argv[1]
    rewrt_flag = bool(argv[2]) if len(argv) == 3 else False # default
    
    dpark_ctx = dpark.DparkContext('mesos')
    def exam_nan(data):
        np_dat = np.array(data)
        n_nan = len(np.where(np_dat != np_dat)[0])
        if n_nan != 0:
            print "Warning, %i numbers nan" % n_nan
    dpark_ctx.beansdb(
            argv[1]
            ).map(
                lambda (key, (val, _, __)): val
            ).foreach(
                exam_nan         
            )
    print 'done'


if __name__ == '__main__':
    main(sys.argv)

    
