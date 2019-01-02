#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import codecs
import os
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from Hyperparams import Hyperparams as hp
from DataLoader import load_test_data, load_src_vocab, load_trgt_vocab
from Graph import *
import pdb
import time

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    src2idx, idx2src = load_src_vocab()
    trgt2idx, idx2trgt = load_trgt_vocab()
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                epoch = len(X) // hp.batch_size
                for i in range(epoch):
                    start = time.time()
                     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2trgt[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("source : " + source +"\n")
                        fout.write("target : " + target + "\n")
                        fout.write("translated : " + got + "\n\n")
                        fout.flush()
                        
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                    batch_time = time.time() - start
                    print("i = {} / {}, time = {}s, remain = {}s".format(i, epoch, batch_time, (epoch-i)*batch_time))

                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
