from warnings import filterwarnings
filterwarnings(action='ignore', category=UserWarning, module='gensim')

from nltk import ngrams
from GoogleVectorsImport import GoogleModel
import numpy as np

# helper function
def make_windows(document, n):
    ''' (str, int) -> list
    Creates ngram windows from a document.
    document is str type; n is int type.
    Returns a list of ngrams.
    '''
    dictionary = {'.':'','?':'','!':''}
    table = str.maketrans(dictionary)
    document = document.translate(table)
    ngram = list(ngrams(document.split(), n))

    windows = []
    for i in range(len(ngram) - n):
        windows.append((ngram[i], ngram[i+n]))

    return windows

# main function
def window_processing_periods(document, n=3, rand=False):
    ''' (str, int, bool) -> tuple
    Makes a tuple of lists. One list holds word2vec representations for two n-grams in a sliding window.
    The other list records if a period lies in between the two n-grams.
    document is text data preprocessed by PreProcessWikiDumps/PreProcessUppsala and DeleteProcessing/ProcessPunctuation .py files.
    n is the size of grams in the sliding window.
    rand allows for uniformly sampled word2vec representations if a word isn't in the GoogleModel. Default is all zeros vector.
    Returns the tuple of lists, ready for a keras model.
    '''
    windows = make_windows(document, n)
    vectors = []
    for w in windows:
        gram1 = ()
        for word in w[0]:
            try:
                gram1 = gram1 + (GoogleModel[word],)
            except KeyError:
                if rand:
                    gram1 = gram1 + (np.random.uniform(-1, 1, (300,)).astype(np.float32),)
                else:
                    gram1 = gram1 + (np.zeros((300,), dtype=np.float32),)
        gram1 = np.array(gram1)

        gram2 = ()
        for word in w[1]:
            try:
                gram2 = gram2 + (GoogleModel[word],)
            except KeyError:
                if rand:
                    gram2 = gram2 + (np.random.uniform(-1, 1, (300,)).astype(np.float32),)
                else:
                    gram2 = gram2 + (np.zeros((300,), dtype=np.float32),)
        gram2 = np.array(gram2)

        vectors.append(np.asarray([gram1, gram2]))

    return np.asarray(vectors) 
