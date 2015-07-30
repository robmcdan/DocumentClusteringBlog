__author__ = 'robm'
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from nltk.stem.snowball import SnowballStemmer
import nltk

'''
tutorial on simple document clustering using sci-kit learn
'''

class DocLoader():
    def __init__(self, document_directory='/Users/robm/Documents/BlogCode/DocumentClusteringBlog/Documents'):
        self.terms = None
        self.tfidf_matrix = None
        self._document_directory = '/Users/robm/Documents/BlogCode/DocumentClusteringBlog/Documents'
        if not os.path.exists(self._document_directory):
            raise Exception('Could not locate document directory: {0}'.format(self._document_directory))

    def vectorize_documents(self, max_df=0.8, min_df=0.2, idf=True, use_stemmer=True):
        tokenizer = None
        if use_stemmer:
            tokenizer = self._stemming_tokenizer
        tfidf_vectorizer = TfidfVectorizer(max_df=max_df,
                                       min_df=min_df, stop_words='english',
                                       use_idf=idf, tokenizer=tokenizer, ngram_range=(1, 3))
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self._document_directory)
        self.terms = tfidf_vectorizer.get_feature_names()

    def _stemming_tokenizer(text):
        stemmer = SnowballStemmer("english")
        tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        #for token in tokens:
        #    if re.search('[a-zA-Z]', token):
        #        filtered_tokens.append(token)
        #stems = [stemmer.stem(t) for t in filtered_tokens]
        stems = [stemmer.stem(t) for t in tokens]
        return stems

if __name__ == "__main__":
    doc_loader = DocLoader()