#! /usr/bin/env python

import os
import sys
import math


def main(argv):
    assert len(argv) == 2 or len(argv) == 3, "<usage>: ./check_nan_tr.py filename rewrt_flag"
    assert os.path.isfile(argv[1])
    print 'Check nan in training file %s' % argv[1]
    rewrt_flag = bool(argv[2]) if len(argv) == 3 else False # default 0, not rewriting.
    fin = open(argv[1])
    if rewrt_flag:
        print 'Will rewrite file purging nan'
        fout = open(argv[1]+'.tmp', 'w')
    data = []
    for i, line in enumerate(fin):
        _line = map(lambda x: float(x), line.split())
        if any(x!=x for x in _line):
            print "Warining! line number %i nan." % i
            continue
        if rewrt_flag:
            fout.write(line)
        #print "line number %i ok" % i
    fin.close()
    if rewrt_flag:
        fout.close()
        os.rename(argv[1]+'.tmp', argv[1])
    print 'done'
    
if __name__ == '__main__':
    main(sys.argv)
