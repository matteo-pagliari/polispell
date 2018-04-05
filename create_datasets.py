# -*- coding: utf8 -*-

import codecs
import os
import errors as er
import re
import string
from nltk import sent_tokenize, word_tokenize

typos_error_param = 0.0025
token_num = '<NUM>'
removed_chars = ["''", "``", u'\u201d', u'\u201c']
sentence_delimiters = r'[.;!?]'
accented_characters = u'èéòàùìÈ'

# Particular characters to substitute
pattern_substitution = [['"', ''], [u'«', ''], [u'»', ''], [u'’', "'"], [u'‘', ''],
                        [u'ľ', "l'"], [u'–', ''], [';', '.']]

wiki_files = '../text/Wiki_corpus/'
num_vers = 1



def create_datasets(files_path):

    files_list = os.listdir(files_path) # Testing
    for filename in files_list:
        with open(os.path.join(files_path, str(filename)), 'r') as f:
            tokens = []
            for text in f:
                text = text.strip()
                # TODO: modify in order to count words not chars
                if len(text) > 20:
                    if not text.startswith('<doc id='):
                        # Maybe it's better to apply substitution for all the text
                        for pattern, substitution in pattern_substitution:
                            text = re.sub(pattern, substitution, text)
                        try:
                            text_sentences = sent_tokenize(text, language='italian')
                        except UnicodeDecodeError:
                            pass
                        if text_sentences:
                            for sent in text_sentences:
                                tokens.append(sent)

        create_clean_corpus_file(tokens, os.path.join('../text/Test', str(filename) + '.text'))


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

    cleaned_corpus = []
    # Consider only sentences with no UNK characters
    for sent in clean_corpus:
        if('<UNK>' not in sent):
            cleaned_corpus.append(sent)

    return cleaned_corpus



def insert_errors_corpus(corpus, sent_error_param):

    errors_corpus = []
    for sentence in corpus:
        sentence = sentence.split(' ')
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
        errors_sentence = errors_sentence.split(' ')

        errors_corpus.append(errors_sentence)

    return errors_corpus



def create_clean_corpus_file(sentences, outfile):

    output= codecs.open(outfile, 'w','utf-8')
    clean = clean_data(sentences)
    for s in clean:
        clean_sent = ' '.join(s)
        output.write(clean_sent + '\n')
    output.close()

    return clean


def create_error_corpus_file(clean_corpus, outfile):

    outfile = codecs.open(outfile, 'w','utf-8')
    err_corpus = insert_errors_corpus(clean_corpus, 0)
    for s in err_corpus:
        err_sent = ' '.join(s)
        outfile.write(err_sent + '\n')
    outfile.close()

    return err_corpus


def create_error_dataset(correctfiles_path, versions):

    for n in range(versions):
        for f in os.listdir(correctfiles_path):
            name = f
            file = codecs.open(os.path.join(correctfiles_path, f), 'r', 'utf-8')
            ita_file = []
            ita_err_file = []
            if(re.search('.text', f)):


                file_sent = file.readlines()
                errors_file = insert_errors_corpus(file_sent, 0)
                # print errors_file[0]
                with codecs.open(os.path.join(correctfiles_path, name), 'r', 'utf-8') as ita:
                    # ita_text = ita.read()
                    for i,e in zip(ita, errors_file):
                        # print i,' '.join(e)
                        if (i != ' '.join(e)):
                            ita_file.append(i)
                            ita_err_file.append(' '.join(e))
                        # print 'len: ', len(e), len(i)

                ita.close()
                # fix saving files
                itafile = codecs.open(os.path.join(correctfiles_path, str(name)[:7]+'.itacor'+'.'+str(n)), 'w','utf-8')
                itaerrfile = codecs.open(os.path.join(correctfiles_path, str(name)[:7]+'.itaerr'+'.'+str(n)), 'w','utf-8')
                print 'len: ', len(ita_file), len(ita_err_file)
                
                for i,e in zip(ita_file, ita_err_file):
                    itafile.write(i)
                    itaerrfile.write(e)
                    
                itafile.close()
                itaerrfile.close()


def create_dataset_stats(corpus_path):

    dir_list = [dir for dir in os.listdir(corpus_path) if os.listdir(os.path.join(corpus_path, dir))]

    for dir in dir_list:
        print dir
        dir_path = os.path.join(corpus_path,dir)
        file_list = os.listdir(dir_path)

        for file in file_list[:1]: # Testing
            print file
            with codecs.open(os.path.join(dir_path, file), 'r', 'utf8') as f:
                plain_text = ''
                for text in f:
                    text = text.strip()
                    if not text.startswith('<doc id=') and not text.startswith('</doc>') and text:
                        plain_text = plain_text + ' ' + text

            for pattern, substitution in pattern_substitution:
                plain_text = re.sub(pattern, substitution, plain_text)
            try:
                text_sentences = sent_tokenize(plain_text, language='italian')
            except UnicodeDecodeError:
                pass

            create_clean_corpus_file(text_sentences, os.path.join('../text/Stats', str(file) + '.text'))





# ---------------------- Run -----------------------------#


create_dataset_stats(wiki_files)
# create_datasets(wiki_files)
# create_error_dataset('../text/Test',2)