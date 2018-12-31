#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

from Hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import pdb

def load_vocab(vocab_file):
    vocab = [line.split()[0] for line in codecs.open(vocab_file, "r","'utf-8").read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_src_vocab():
    return load_vocab(hp.source_vocab)

def load_trgt_vocab():
    return load_vocab(hp.target_vocab)

def create_data(source_sents, target_sents): 
    src2idx, idx2src = load_src_vocab()
    trgt2idx, idx2trgt = load_trgt_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [src2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [trgt2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], "constant", constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], "constant", constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    src_sents = [regex.sub("[^\s\p{Han}']", "", line) for line in codecs.open(hp.source_train, "r", "utf-8").read().split("\n") if line and line[0] != "<"]
    trgt_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, "r", "utf-8").read().split("\n") if line and line[0] != "<"]
    X, Y, Sources, Targets = create_data(src_sents, trgt_sents)
    return X, Y
    
def load_test_data():
#     def _refine1(line):
#         line = regex.sub("[^\s\p{Han}']", "", line) 
#         return line.strip()
# 
#     def _refine2(line):
#         line = regex.sub("[^\s\p{Latin}']", "", line) 
#         return line.strip()
#     
#     src_sents = [_refine1(line) for line in codecs.open(hp.source_test, "r", "utf-8").read().split("\n")]
#     trgt_sents = [_refine2(line) for line in codecs.open(hp.target_test, "r", "utf-8").read().split("\n")]
    src_sents = [regex.sub("[^\s\p{Han}']", "", line).strip() for line in codecs.open(hp.source_test, "r", "utf-8").read().split("\n")]
    trgt_sents = [regex.sub("[^\s\p{Latin}']", "", line).strip() for line in codecs.open(hp.target_test, "r", "utf-8").read().split("\n")]
        
    X, Y, Sources, Targets = create_data(src_sents, trgt_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

