"""
This module implement the LSA class, which will take reponsible for processing
text and compare two documents using the latent semantic analysis algorithm.
"""
from typing import Optional
import string
import collections
import numpy as np

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

def make_vocab(texts: str, num_vocab: Optional[int] = None, min_word_count: Optional[int] = None) -> None:
    """Create a vocab mapping {term: index}}
    Args:
        texts: one string that contains all the text data
        num_vocab: the desired number of vocabulary
        min_word_count: if specified, the number of vocab will be infered such 
        that every vocabulary appear at least min_word_count times int the texts
    """

    if num_vocab is None and min_word_count is None:
        raise ValueError('Either num_vocab or min_word_count must be provided')

    texts = texts.lower()
    # remove puntuations
    texts = texts.translate(str.maketrans('', '', string.punctuation))
    
    texts = texts.split()
    # remove stopwords
    texts = [x for x in texts if x not in stop_words]
    # lemmatize words
    lemma = nltk.wordnet.WordNetLemmatizer()
    texts = [lemma.lemmatize(x) for x in texts]
    # takes num_vocab most common words
    counter = collections.Counter(texts)

    selected_words = []
    if min_word_count is None and num_vocab is not None:
        selected_words = counter.most_common(num_vocab)
        selected_words = [x[0] for x in selected_words]
    elif min_word_count is not None:
        word_count = counter.most_common(len(texts))
        selected_words = [x[0] for x in word_count if x[1] >= min_word_count]

    
    vocab = {w: i+1 for i, w in enumerate(selected_words)}
    vocab['_unknown_'] = 0
    
    return vocab


if __name__ == '__main__':
    texts = 'The cat sat on the mat with another cat sat on the floor and another cat sat on the table'
    print(make_vocab(texts, min_word_count = 2))