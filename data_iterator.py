'''
Classes that allow to iterate over dataset made by multiple files
'''


import utils
import os
from utils import load_dict


class BiTextIterator:

    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=6, maxlen=100,
                 n_words_source=-1, n_words_target=-1,
                 skip_empty=False, shuffle_each_epoch=False,
                 sort_by_length=True, maxibatch_size=1):

        self.source = open(source, 'r')
        self.target = open(target, 'r')

        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        # Consider only num_encoder_symbols
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False


    def __iter__(self):
        return self


    def __len__(self):
        return sum([1 for _ in self])


    def reset(self):

        self.source.seek(0)
        self.target.seek(0)
        self.source_buffer = []
        self.target_buffer = []


    def next(self):

        '''
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        '''

        source = []
        target = []

        assert len(self.source_buffer) == len(self.target_buffer) # Buffer size mismatch!!!

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                tt = self.target.readline()

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            if len(self.source_buffer) == 0:
                self.end_of_data = True
                raise StopIteration

        # Exit condition for the iteration
        if all(not elem for elem in self.source_buffer):
            self.end_of_data = True
            self.reset()
            raise StopIteration


        try:
            while True:

                try:
                    # Every cycle it removes a batch
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                ss = [self.source_dict[w] if w in self.source_dict
                      else utils.unk_token for w in ss]

                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict
                      else utils.unk_token for w in tt]

                source.append(ss)
                target.append(tt)


        except IOError:
            self.end_of_data = True


        return source, target



class TextIterator:

    def __init__(self, source, source_dict,
                 batch_size=5, maxlen=100,
                 n_words_source=-1,
                 skip_empty=False, shuffle_each_epoch=False,
                 sort_by_length=True, maxibatch_size=1):

        self.source = open(source, 'r')
        self.source_dict = load_dict(source_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self


    def __len__(self):
        return sum([1 for _ in self])


    def reset(self):

        self.source.seek(0)
        self.source_buffer = []


    def next(self):

        source = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                self.source_buffer.append(ss.strip().split())

            if len(self.source_buffer) == 0:
                self.end_of_data = True
                raise StopIteration

        # Exit condition for the iteration
        if all(not elem for elem in self.source_buffer):
            self.end_of_data = True
            self.reset()
            raise StopIteration

        try:
            while True:

                try:
                    # Every cycle it removes a batch
                    ss = self.source_buffer.pop()
                    # print len(ss)
                except IndexError:
                    break

                ss = [self.source_dict[w] if w in self.source_dict
                      else utils.unk_token for w in ss]

                source.append(ss)


        except IOError:
            print 'end_of_data'
            self.end_of_data = True


        return source


class CorpusIterator:

    def __init__(self, sourcepath, batch_size, source_vocabulary):

        self.sourcepath = sourcepath
        self.source_list = os.listdir(sourcepath)
        self.source_id = 0

        self.batch_size = batch_size
        self.source_vocabulary = source_vocabulary


    def __iter__(self):
        return self


    def __len__(self):
        return sum([1 for _ in self])


    def next(self):

        if(self.source_id >= len(self.source_list)):
            raise StopIteration

        source_text = os.path.join(self.sourcepath,self.source_list[self.source_id])
        iterator = TextIterator(source=source_text, batch_size=self.batch_size,
                            source_dict=self.source_vocabulary, maxlen=None,
                            n_words_source=30000)

        self.source_id +=1

        return iterator

# TODO: class CorpusBiTextIterator