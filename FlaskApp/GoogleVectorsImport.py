from os import walk
from os.path import expanduser
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

for root, dirs, paths in walk(expanduser('~')):
    if 'GoogleNews-vectors-negative300.bin' in paths:
        file = root + '\\' + paths[paths.index('GoogleNews-vectors-negative300.bin')]

GoogleModel = KeyedVectors.load_word2vec_format(file, binary=True)
