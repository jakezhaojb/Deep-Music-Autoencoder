#!/bin/env python

from yaafelib import FeaturePlan, Engine, AudioFileProcessor
from sliding_window import sliding_window


def compute_aggr_spec_seq(spec_mat, win_size, hop_size):
    shp2 = spec_mat.shape[1]
    mat_aggr = []
    for sub_mat in sliding_window(spec_mat, (win_size, shp2), (hop_size, shp2)):
        aggre_vec = sub_mat.mean(axis=0)
        mat_aggr.append(aggre_vec.tolist())
    return mat_aggr


def compute_spec(song_path):
    fp = FeaturePlan(sample_rate=22050, resample=True)
    #add one feature
    fp.addFeature("spec: PowerSpectrum blockSize=1024 stepSize=512")
    df = fp.getDataFlow()
    engine = Engine()
    engine.load(df)
    afp = AudioFileProcessor()
    afp.processFile(engine, song_path)
    return engine.readOutput('spec')


def process_mp3(song_path):
    spec_mat = compute_spec(song_path)
    win_size = 64  # int(22050. / 512 * 1.0) ~ 1 s ~ 43 p.t.s
    hop_size = win_size / 2
    new_mat = compute_aggr_spec_seq(spec_mat, win_size, hop_size)
    return new_mat
