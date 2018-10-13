'''
Seth Temple
CIS 401 Research
Place the Period Problem
Import Vector Representations
'''

import os
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

file_path = '/projects/fickaslab/stemple/pythons/'
file_name = 'GoogleNews-vectors-negative300.bin'
file = file_path + file_name

GoogleModel = KeyedVectors.load_word2vec_format(file, binary=True)