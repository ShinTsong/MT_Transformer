#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

class Hyperparams:
    # data
    source_file = "corpora/cn.txt"
    target_file = "corpora/en.txt"

    source_train = "corpora/train_cn"
    target_train = "corpora/train_en"
    source_test = "corpora/test_cn"
    target_test = "corpora/test_en"
    
    source_vocab = "corpora/cn_vocab"
    target_vocab = "corpora/en_vocab"

    # training
    batch_size = 32
    lr = 0.0001
    logdir = "logdir"
    
    # model
    maxlen = 100
    min_cnt = 20
    hidden_units = 512
    num_blocks = 6
    num_epochs = 60
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False 
