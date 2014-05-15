# coding: utf-8
BDB_DIR = '/mfs/user/panmiaocai/projects/sparse_autoencoder/dataset'\
          '/output_spec_vec_small'

import dpark


def main():
    dpark_ctx = dpark.DparkContext()
    vec_rdd = dpark_ctx.beansdb(
        BDB_DIR
    ).map(
        lambda (key, (val, _, __)): val
    )

    data = vec_rdd.collect()
    print len(data)
    print len(data[0])


if __name__ == '__main__':
    main()
