#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from collections import Counter

import argparse
import os, codecs
import random
import pdb
import time

from Hyperparams import Hyperparams as hp
from DataLoader import get_batch_data, load_src_vocab, load_trgt_vocab
from Graph import *

def make_vocab(fpath, fname):
    text = codecs.open(fpath, "r", "utf-8").read()
    words = text.split()
    word2cnt = Counter(words)
    with codecs.open(fname, "w", "utf-8") as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

def split_dataset(src_file, trgt_file, src_train, trgt_train, src_test, trgt_test, ratio):
    with codecs.open(src_file, "r", "utf-8") as src, codecs.open(trgt_file, "r", "utf-8") as trgt:
        src_texts = src.readlines()
        trgt_texts = trgt.readlines()

    length = len(src_texts)
    num = int(length * ratio)

    train_idx = random.sample(range(length), num)
    max_src_len = 0
    max_trgt_len = 0
    with codecs.open(src_train, "w", "utf-8") as train_src, codecs.open(trgt_train, "w", "utf-8") as train_trgt, codecs.open(src_test, "w", "utf-8") as test_src, codecs.open(trgt_test, "w", "utf-8") as test_trgt:
        for i in range(length):
            if i in train_idx:
                train_src.write(src_texts[i])
                train_trgt.write(trgt_texts[i])
            else:
                test_src.write(src_texts[i])
                test_trgt.write(trgt_texts[i])
            
            if len(src_texts[i]) > max_src_len:
                max_src_len = len(src_texts[i])
                
            if len(trgt_texts[i]) > max_trgt_len:
                max_trgt_len = len(trgt_texts[i])
                
    print(max_src_len, max_trgt_len)

def preprocess():
    split_dataset(hp.source_file, hp.target_file, hp.source_train, hp.target_train, hp.source_test, hp.target_test, ratio=0.8)
    make_vocab(hp.source_train, hp.source_vocab)
    make_vocab(hp.target_train, hp.target_vocab)
    print("Preprocessed!")

def train():
    # Load vocabulary    
    src2idx, idx2src = load_src_vocab()
    trgt2idx, idx2trgt = load_trgt_vocab()
    
    # Construct graph
    g = Graph("train"); 
    print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, logdir=hp.logdir, save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
            print('/model_epoch_%02d_gs_%d' % (epoch, gs))
    print("Trained!")    

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
                print("Bleu Score = {}\n".format(str(100*score)))
    print("Evaluated!")
                                          
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, default="all", help="mode : preprocess, train, eval, all(defualt)")
    args = parser.parse_args()

    if args.mode == "all":
        preprocess()
        train()
        eval()
    elif args.mode == "preprocess":
        preprocess()
    elif args.mode == "train":
        train()
    elif args.mode == "eval":
        eval()
    else:
        print("param error!")
