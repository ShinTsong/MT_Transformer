#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow as tf
from Hyperparams import Hyperparams as hp
from DataLoader import get_batch_data, load_src_vocab, load_trgt_vocab
from Graph import *
from tqdm import tqdm
import os, codecs

if __name__ == '__main__':                
    # Load vocabulary    
    src2idx, idx2src = load_src_vocab()
    trgt2idx, idx2trgt = load_trgt_vocab()
    
    # Construct graph
    g = Graph("train"); 
    print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
            print('/model_epoch_%02d_gs_%d' % (epoch, gs))
    print("Done")    
