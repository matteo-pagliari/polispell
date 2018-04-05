# -*- coding: utf8 -*-
import utils
from hyphen import Hyphenator

class hyphenator:

    def __init__(self,language='it_IT'):

        self.h = Hyphenator(language)

    def split_syllables(self, word):

        syllables = self.h.syllables(utils.check_unicode(word))

        return syllables

    def split_word(self,word):
        
        pairs = self.h.pairs(utils.check_unicode(word))

        return pairs
