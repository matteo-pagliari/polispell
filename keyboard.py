# -*- coding: utf8 -*-
import numpy as np
import string
import sys
reload(sys)
sys.setdefaultencoding('Cp1252')

class Keyboard:

    def __init__(self):

        self.key = np.zeros(shape=(3,12), dtype='string')
        self.key[0] = ['q','w','e','r','t','y','u','i','o','p','è','0']
        self.key[1] = ['a','s','d','f','g','h','j','k','l','ò','à','ù']
        self.key[2] = ['z','x','c','v','b','n','m',',','.','0','0','0']

        self.pos = [[-1,-1],[0,-1],[-1,0],[0,1],[1,1],[1,-1],[-1,1],[1,0]]


    def nearest(self,char):

        # To find all the possible characters in the keyboard
        char = char.lower()
        # Now I don't wanna make errors with punctuaction or similar
        if(char in string.letters):
            i,j = np.where(self.key == char)
            list_idx = [int(i),int(j)]
            nearest = []
            for idx in self.pos:
                near = np.array(list_idx) + np.array(idx)
                near = np.array(near,dtype=int)
                if(near[0]>=0 and near[1]>=0 and near[0]<=2 and near[1]<=11):
                    elem = self.key.item((near[0],near[1]))
                    if(elem !='0'):
                        nearest.append(elem)
        else:
            nearest = char

        return nearest



