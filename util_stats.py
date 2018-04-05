# -*- coding: utf8 -*-

import codecs
import os
import numpy as np
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt

stats_path = '../text/Stats'

# Printing histogram
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 40}

plt.rc('font', **font)
plt.figure(figsize=(50,50))


def length_sentences_stats(corpus_path):

    '''
    dir_list = [dir for dir in os.listdir(corpus_path) if os.listdir(os.path.join(corpus_path, dir))]

    for dir in dir_list:
        print dir
        dir_path = os.path.join(corpus_path, dir)
    '''

    file_list = os.listdir(corpus_path)
    lengths = []
    for file in file_list:
        print file
        with codecs.open(os.path.join(corpus_path, file), 'r', 'utf8') as f:
            text = f.readlines()
            for line in text:
                line = line.split()
                if len(line) != 0:
                    lengths.append(len(line))

    avg = sum(lengths)/len(lengths)

    sorted_lengths = sorted(lengths, key=int)

    counter = Counter(sorted_lengths)

    dict = OrderedDict(counter)

    plt.bar(dict.keys(), dict.values(), color='b')
    plt.title("Words Lengths")
    # plt.xticks(np.arange(min(elem), max(elem) + 100, 100.0))
    # plt.yticks(np.arange(0, max(sort_word_freqs) + 1000, 1000.0))
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.savefig('../text/Statslength_freq.pdf')
    plt.show()

    return  avg, max(lengths),  min(lengths), np.var(lengths), counter


def words_stats(corpus_path):

    file_list = os.listdir(corpus_path)
    word_freqs = OrderedDict()
    for file in file_list:
        with codecs.open(os.path.join(corpus_path,file), 'r', 'utf8') as file:
            for line in file:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    sort_word_freqs = sorted(word_freqs.values(), key=int, reverse=True)[:1000] # Testing
    elem = range(len(sort_word_freqs))

    plt.bar(elem, sort_word_freqs, width=10)
    plt.title("Words Freqs")
    plt.xticks(np.arange(min(elem), max(elem) + 100, 100.0))
    plt.yticks(np.arange(0, max(sort_word_freqs) + 1000, 1000.0))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig('../text/Stats/word_freq.pdf')

    return word_freqs, max(sort_word_freqs), min(sort_word_freqs)

