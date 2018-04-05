# -*- coding: utf8 -*-
'''
Functions that are useful to create train, validation and test set
and also for creating dictionaries
'''

import re
import codecs
import json
import numpy as np
from random import choice
from collections import Counter, OrderedDict


# Extra vocabulary symbols
# Check if NUM is correctly inserted in the vocabulary
GO = '_GO'
EOS = '_EOS'
UNK = '_UNK'
NUM = '_NUM'

extra_tokens = [GO, EOS, UNK, NUM]

start_token = extra_tokens.index(GO)
end_token = extra_tokens.index(EOS)
unk_token = extra_tokens.index(UNK)
num_token = extra_tokens.index(NUM)


def build_dictionary(filenamepath, name):

    word_freqs = OrderedDict()
    with codecs.open(filenamepath, 'r', 'utf8') as file:
        for line in file:
            words_in = line.strip().split(' ')
            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w]+=1


    with open('../text/%s.json'%name, 'wb') as f:
        json.dump(word_freqs, f, ensure_ascii=True)


def build_dictionary_unique_words(filenamepath, name, most_common):

    text_file = codecs.open(filenamepath, 'r', 'utf8')
    read_file = text_file.read()
    word_list = read_file.split()

    most_common_words = dict(Counter(word_list).most_common(most_common-len(extra_tokens)))

    uniquewords = most_common_words.keys()

    uniquewords = extra_tokens + uniquewords
    unique_dict = OrderedDict(zip(uniquewords,range(0,len(uniquewords))))

    with open('../text/%s.json'%name, 'wb') as f:
        json.dump(unique_dict, f, ensure_ascii=True)


def create_mini_corpus(filepath, outfilepath, maxline, minline=0):

    with codecs.open(outfilepath, 'wb', 'utf8') as mini:
        with codecs.open(filepath, 'r', 'utf8') as f:
            l = 0
            for line in f:
                if(l>=maxline):
                    break
                if(l>=minline):
                    mini.write(line)
                l+=1
    f.close()
    mini.close()



# ------------------------Run---------------------------#



# build_dictionary_unique_words('../text/ita_err.txt', 'ita_err.unique')
# build_dictionary_unique_words('../text/ita.txt', 'ita.unique')



'''
for i in range(3):
    create_mini_corpus('../text/ita_err.txt', '../text/validation/ita_err.mini.test_'+str(i), maxline=30*(i+1), minline=i*30)

for i in range(3):
    create_mini_corpus('../text/ita_err.txt', '../text/train/ita_err.mini'+str(i), maxline=30*(i+1), minline=i*30)
    create_mini_corpus('../text/ita.txt', '../text/train/ita.mini'+str(i), maxline=30*(i+1), minline=i*30)
'''

