#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

from Hyperparams import Hyperparams as hp
from collections import Counter
import codecs
import random

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

if __name__ == '__main__':
    split_dataset(hp.source_file, hp.target_file, hp.source_train, hp.target_train, hp.source_test, hp.target_test, ratio=0.8)
    make_vocab(hp.source_train, hp.source_vocab)
    make_vocab(hp.target_train, hp.target_vocab)
    print("Done")
