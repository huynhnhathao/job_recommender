from typing import Optional, Dict, Union, List, Any
import string
import collections
import logging
import json
import os

import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import text 
from sklearn import decomposition


import constants

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

logger = logging.getLogger()

def is_ascii(s: str) -> bool:
    """Check if a string s only contains ascii chartecter"""
    return all(ord(c) < 128 for c in s)

def contains_digit(text: str) -> bool:
    """Check if a string contains a digit"""
    return any(x.isdigit() for x in text)

def make_vocab(texts: str, num_vocab: Optional[int] = None,
                min_word_count: Optional[int] = None) -> None:
    """Create a vocab mapping {term: index}}
    Args:
        texts: one string that contains all the text data
        num_vocab: the desired number of vocabulary
        min_word_count: if specified, the number of vocab will be infered such 
        that every vocabulary appear at least min_word_count times int the texts
    """
    # load saved vocab if exist
    if os.path.isfile(constants.VOCAB_PATH):
        logger.info(f'Loading vocab from {constants.VOCAB_PATH}')
        with open(constants.VOCAB_PATH, 'r') as f:
            vocab = json.load(f)
        return vocab

    if num_vocab is None and min_word_count is None:
        raise ValueError('Either num_vocab or min_word_count must be provided')

    texts = texts.lower()
    # remove puntuations
    texts = texts.translate(str.maketrans('', '', string.punctuation))
    
    texts = texts.split()
    # remove stopwords, digits, punctuations(again) and word that have non-ascii
    # character
    texts = [x for x in texts if x not in stop_words and x not in string.punctuation and is_ascii(x) and not contains_digit(x)]
    # lemmatize words
    lemma = nltk.wordnet.WordNetLemmatizer()
    texts = [lemma.lemmatize(x) for x in texts]
    # takes num_vocab most common words, but don't accept number and non-English words

    counter = collections.Counter(texts)

    selected_words = []
    if min_word_count is None and num_vocab is not None:
        selected_words = counter.most_common(num_vocab)
        selected_words = [x[0] for x in selected_words if not x[0].isdigit()]
    elif min_word_count is not None:
        word_count = counter.most_common(len(texts))
        selected_words = [x[0] for x in word_count if x[1] >= min_word_count and not x[0].isdigit() ]

    
    vocab = {w: i+1 for i, w in enumerate(selected_words)}
    vocab['_unknown_'] = 0
    
    logger.info(f'Saving vocab to {constants.VOCAB_PATH}')
    with open(constants.VOCAB_PATH, 'w') as f:
        json.dump(vocab, f)

    return vocab

class LSA:

    def __init__(self, vocab: Dict[str, int], documents: List[str],
                num_features: Union[int, float]) -> None:
        """
        This class's instance can represent text using latent semantic analysis
        and compare two text using Cosine similarity.
        
        Args:
            vocab: a dictionary map from term to index
            documents: a list of all documents used to fit the tf-idf vectorizer
            num_features: number of features after dimension reduction. If it is
                a float, the number of reduced features will be
                num_features*len(vocab)
        """
        logger.info('Initializing LSA comparer...')
        self.vocab = vocab
        self.documents = documents
        self.num_features = num_features if num_features > 1 else int(num_features*len(self.vocab))
        
        self.lemmatizer = nltk.wordnet.WordNetLemmatizer()

        self.vectorizer = text.TfidfVectorizer(decode_error='replace', 
                            vocabulary=self.vocab, )
        self.svd = decomposition.TruncatedSVD(n_components=self.num_features,
                            random_state = 42)

    def preprocess_text(self, text: str) -> str:
        """Preprocess a given text, replace out-of-vocab words with '_unknown_' 
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.split()
        text = [x for x in text if x not in stop_words and x not in string.punctuation and is_ascii(x) and not contains_digit(x)]
        text = [self.lemmatizer.lemmatize(x) for x in text]
        # replace oov word with '_unknown_'
        text = [x if x in self.vocab.keys() else '_unknown_' for x in text ]
        return ' '.join(text)

    def do_work(self) -> None:
        """Preprocess all documents and fit the tf-idf vectorizer and the
        SVD features reducer
        """
        logger.info('Fitting vectorizer and features reducer...')
        self.processed_documents = [self.preprocess_text(x) for x in self.documents]
        features_matrix = self.vectorizer.fit_transform(self.processed_documents)
        self.svd.fit(features_matrix)

        logger.info('Done.')

    def vectorize(self, document: str) -> np.ndarray:
        """Vectorize a document.
        """
        
        document = self.preprocess_text(document)
        tfidf = self.vectorizer.transform([document],)
        reduced_features = self.svd.transform(tfidf)
        return reduced_features

    def compare_two_text(self, text1: str, text2: str,
                        method: str= 'Cosine')-> float:
        """This method takes two text and perform preprocess, vectorizer, reduce
        features and compare two vector using the specified method
        This method will be used to compute the 'profile match' relation between
        a candidate and a job.
        """
        pass


    def compute_linear_kernel_matrix(self, documents: List[str]) -> np.ndarray:
        """Compute and return the linear kernel matrix of all the documents using
        latent semantic analysys.
        The returned matrix is symetric and is an unscaled version of the Cosine similarity
        matrix.
        This method will be used to compute the 'similar' relation between entities
        of the same type.
        """
        pass

if __name__ == '__main__':
    texts = 'The cat sat on the mat with another cat sat on the floor and another cat sat on the table'
    print(make_vocab(texts, min_word_count = 2))