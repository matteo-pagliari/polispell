# -*- coding: utf8 -*-
import numpy as np
import random
import unidecode
from keyboard import Keyboard
from hyphenator import hyphenator


kb = Keyboard()
hp = hyphenator()

similar_phonemes = [['a','e'],['p','b'],['f','v'],['a','u'],['t','d'],['cu','qu'],['c','l'],['u','v'],['ni','ci'],
           ['d','b'],['gn','gl'],['p','v']]
similar_graphemes = [['p','d'],['b','q'],['m','n'],['c','e'],['s','z']]


def insert_typos(word, error_param):

    # Insert random character
    if np.random.rand() < len(word) * error_param:
        ins_position = np.random.randint(0,len(word))
        rand_char = random.choice(kb.nearest(word[ins_position]))
        # rand_char = random.choice(string.letters)
        word = word[:ins_position] + rand_char + word[ins_position:]

    # Delete random character
    if ((np.random.rand() < len(word) * error_param) and len(word)>2):
        del_position = np.random.randint(0,len(word))
        word = word[:del_position] + word[del_position+1:]

    # Transpose 2 close characters
    if np.random.rand() < len(word) * error_param:
        if(len(word)!=1):
            tran_position = np.random.randint(0,len(word)-1)
            word = word[:tran_position] + word[tran_position+1] + word[tran_position] + word[tran_position+2:]

    # Swap character with a random character
    if np.random.rand() < len(word) * error_param:
        swp_position = np.random.randint(0,len(word))
        rand_char = random.choice(kb.nearest(word[swp_position]))
        # rand_char = random.choice(string.letters)
        word = word[:swp_position] + rand_char + word[swp_position+1:]

    return word


def insert_dyslexia_errors(word, error_param):

    # Split words in 2 parts
    # TODO: check if it's possible to remove last condition
    if ((np.random.rand() < len(word) * error_param) and len(word)>4 and "'" not in word): # problem with single quote if I want to split word
        # Catch IndexError in case the word cannot be splitted by the hyphenator
        try:
            pairs = hp.split_word(word)
        except IndexError:
            print 'IndexError', word
        else:
            try:
                word = random.choice(pairs)
            except IndexError:
                print 'IndexError', word
            else:
                word = word[0] + ' ' + word[1]

    # Remove accents from word
    if np.random.rand() < len(word) * error_param:
        word = unidecode.unidecode(word)

    # Remove 'h' from word
    if np.random.rand() < len(word) * error_param:
        word = word.replace("h","",1)

    # Repeat syllables (len(syl)==2)
    if (np.random.rand() < len(word) * error_param) and len(word)>5:
        syls = hp.split_syllables(word)
        # For ords that cannot be splitted
        if(len(syls)>0):
            double_idx = np.random.randint(0,len(syls))
            double = syls[double_idx]
            if len(double)==2:
                word = ''.join(str(l) for l in syls[:double_idx]) + double + ''.join(str(l) for l in syls[double_idx:])

    # Swap consonant-vowel
    if (np.random.rand() < len(word) * error_param) and len(word)>=4:
        syls = hp.split_syllables(word)
        if(len(syls)>=2):
            syl1 = syls[0]
            syl2 = syls[1]
            if(len(syl1)==2 and len(syl2)==2):
                word = syl2[0] + syl1[1] + syl1[0] + syl2[1] + ''.join(str(l) for l in syls[2:])

    # Swap similar phonemes
    if (np.random.rand() < len(word) * error_param):
        swp = random.choice(similar_phonemes)
        if(swp[0] in word):
            word = word.replace(swp[0],swp[1])
        elif(swp[1] in word):
            word = word.replace(swp[1],swp[0])

    # Swap similar graphemes
    if (np.random.rand() < len(word) * error_param):
        swp = random.choice(similar_graphemes)
        if(swp[0] in word):
            word = word.replace(swp[0],swp[1])
        elif(swp[1] in word):
            word = word.replace(swp[1],swp[0])

    # Move final part of a word to the beginning of the next one
    if (np.random.rand() < len(word) * error_param):
        syls = hp.split_syllables(word)
        if(len(syls)>=2):
            last = syls[-1]
            word = ''.join(str(l) for l in syls[:-1]) + ' ' + last + '-'

    return word

