# -*- coding: utf8 -*-
'''
Functions useful for preparing data for training and decoding
'''

import re
import codecs
import json
import numpy as np
from random import choice
from collections import Counter,OrderedDict


# Extra vocabulary symbols
GO = '_GO'
EOS = '_EOS'
UNK = '_UNK'

extra_tokens = [GO, EOS, UNK]

start_token = extra_tokens.index(GO)
end_token = extra_tokens.index(EOS)
unk_token = extra_tokens.index(UNK)


def check_unicode(string):
    """Check and return unicode string.

    :param string
    :type string: string

    """
    if isinstance(string, unicode):
        temp_string = string
    else:
        temp_string = unicode(string)
    return temp_string


def replace_random(src, frm, to):
    matches = list(re.finditer(frm, src))
    replace = choice(matches)
    return src[:replace.start()] + to + src[replace.end():]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def is_ascii(text):
    if isinstance(text, unicode):
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            return False
    else:
        try:
            text.decode('ascii')
        except UnicodeDecodeError:
            return False
    return True



def load_dict(filename):

    with open(filename, 'rb') as f:
        return dict((key, value ) for (key,value) in json.load(f).items())


def load_inverse_dict(dict_path):

    original_dict = load_dict(dict_path)
    inverse_dict = {}
    # for word, idx in original_dict.iteritems():
    #     inverse_dict[idx] = word
    inverse_dict = {v: k for k, v in original_dict.iteritems()}

    #with open('../text/inverse.json', 'wb') as f:
    #    json.dump(inverse_dict, f, ensure_ascii=True)

    return inverse_dict


def prepare_train_batch(seqs_x_i, seqs_y_i, maxlen=None):

    # Remove empty element
    seqs_x = []
    seqs_y = []
    for x,y in zip(seqs_x_i,seqs_y_i):
        if x and y:
            seqs_x.append(x)
            seqs_y.append(y)

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token
    y = np.ones((batch_size, maxlen_y)).astype('int32') * end_token

    for idx, [s_x, s_y] in enumerate(zip(seqs_x,seqs_y)):
        x[idx, : lengths_x[idx]] = s_x
        y[idx, : lengths_y[idx]] = s_y

    return x, x_lengths, y, y_lengths


def prepare_batch(seqs_x_i, maxlen=None):

    seqs_x = []
    for x in seqs_x_i:
        if x:
            seqs_x.append(x)

    lengths_x = [len(s) for s in seqs_x]

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token

    for idx, s_x in enumerate(seqs_x):
        x[idx, : lengths_x[idx]] = s_x

    return x, x_lengths



def seq2words(seq, inverse_target_dictionary):

    words = []
    for w in seq:
        if w == end_token:
            break
        # print w
        if w[0] in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w[0]])
        else:
            words.append(UNK)

    return ' '.join(words)