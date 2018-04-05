# -*- coding: utf8 -*-

import codecs
import os
import errors as er
import re
import string
from nltk import sent_tokenize, word_tokenize
from collections import Counter

outfilePath = '../text/Test'
typos_error_param = 0.0025
token_num = '<NUM>'
removed_chars = ["''", "``", u'\u201d', u'\u201c']
sentence_delimiters = r'[.;!?]'
accented_characters = u'èéòàùìÈ'

dir = '../text/AA'
plain_suffix = '.plain'
cleaned_suffix = '.cleaned'

# Pattern to substitute
pattern_substitution = []


def create_plain_text_files(corpora_path):
    for filename in os.listdir(corpora_path)[:10]:
        outfile = codecs.open(os.path.join(outfilePath, str(filename) + plain_suffix), 'w')
        # Check str for dir name
        with open(os.path.join(corpora_path, str(filename)), 'r') as f:
            for text in f:
                text = text.strip()
                # TODO: modify in order to count words not chars
                if len(text) > 20:
                    if not text.startswith('<doc id='):
                        outfile.write(text + '\n')
        outfile.close()

create_plain_text_files('../text/AA')


def clean_corpora(corpora_path):

    for filename in os.listdir(corpora_path):
        file = codecs.open(filename, 'rt', 'utf8')
        text = file.read()
        for pattern,substitution in pattern_substitution:
            text = re.sub(pattern, substitution, text)
        file.close()
        sentences = sent_tokenize(text, language='italian')
        clean_corpus = clean_data(sentences)
        create_clean_corpus_file(clean_corpus, str(filename) + '/' + cleaned_suffix)


def load_sentences_test():

    filenamepath = '../text/cleantext_i.txt'
    file = open(filenamepath, 'rt')
    text = file.read()
    text = text.decode('utf8')
    file.close()
    # TODO: try to modify in a better way
    text = re.sub('"', '', text)
    text = re.sub(u'«', '', text)
    text = re.sub(u'»', '', text)
    text = re.sub(u'’', "'", text)
    text = re.sub(u'‘', '', text)
    text = re.sub(u'ľ', "l'", text)
    text = re.sub(u'–', '', text)
    text = re.sub(';', '.', text)
    tokens = sent_tokenize(text,language='italian')
    # TODO: remove also parenthesis
    # tokens = re.split(r'[.;!?]', text)

    return tokens


def clean_data(corpus):

    clean_corpus = []
    for sentence in corpus:
        words = word_tokenize(sentence, language='italian')
        clean_sentence = []
        for word in words:
            if word not in string.punctuation and word not in removed_chars:
                if(re.search(r'[0123456789]',word)):
                #if (is_number(word)):
                    clean_sentence.append('<NUM>')
                elif (all(c in string.ascii_letters + accented_characters + "'" for c in word)):
                    word = word.lower()
                    if ("'" in word):
                        # If a word is in form article + ' + name split in the correct way: article' + name
                        word_with_hyp = re.split("'",word)
                        clean_sentence.append(word_with_hyp[0] + "'")
                        clean_sentence.append(word_with_hyp[1])
                    else:
                        clean_sentence.append(word)
                elif("'" in word):
                    # TODO: split without removing delimiters
                    word_with_hyp = re.split("'",word)
                    clean_sentence.append(word_with_hyp[0]+ "'")
                    clean_sentence.append('<UNK>')
                else:
                    clean_sentence.append('<UNK>')
        clean_corpus.append(clean_sentence)

    return clean_corpus


def insert_errors_corpus(corpus, sent_error_param):

    errors_corpus = []
    for sentence in corpus:
        errors_sentence = []
        for word in sentence:
            if ((word not in string.punctuation)):
                if (word == token_num):
                    errors_sentence.append('<NUM>')
                elif (word == '<UNK>'):
                    # TODO: fix <UNK>
                    errors_sentence.append('<UNK>')
                elif (len(word) > 2):
                    er_typos_word = (er.insert_typos(word, typos_error_param))
                    errors_sentence.append(er.insert_dyslexia_errors(er_typos_word, typos_error_param))
                else:
                    errors_sentence.append(word)

        # Merge syllables
        errors_sentence = ' '.join(errors_sentence)
        errors_sentence = errors_sentence.replace('- ', '')
        errors_sentence = errors_sentence.replace('-', '')
        errors_sentence = errors_sentence.split()

        errors_corpus.append(errors_sentence)

    return errors_corpus


def most_frequent_V_words(corpus,V):

    flat_corpus = [item for sublist in corpus for item in sublist]
    cnt = Counter(flat_corpus)

    return cnt.most_common(V)


def create_clean_corpus_file(sentences, outfile):

    outfile = codecs.open('../text/ita.txt', 'w','utf-8')
    clean = clean_data(sentences)
    for s in clean:
        clean_sent = ' '.join(s)
        outfile.write(clean_sent + '\n')
    outfile.close()

    return clean

def create_error_corpus_file(clean_corpus, outfile):

    outfile = codecs.open('../text/ita_err.txt', 'w','utf-8')
    err_corpus = insert_errors_corpus(clean_corpus, 0)
    for s in err_corpus:
        err_sent = ' '.join(s)
        outfile.write(err_sent + '\n')
    outfile.close()

    return err_corpus


def create_vocab_file(corpus,most_frequen_words,outfile):

    vocab = most_frequent_V_words(corpus,most_frequen_words)
    vocab_file = codecs.open(outfile,'w','utf8')
    for key in vocab:
        vocab_file.write(key[0] + '\n')
    vocab_file.close()



# ------------------Test---------------------------

#s = load_sentences_test()
#ita_vocab = most_frequent_V_words(clean,100)
#ita_err_vocab = most_frequent_V_words(err_corpus,100)

# -------------------------------------------------






