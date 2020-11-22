import nltk
from functools import lru_cache
nltk.download('punkt')

class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=100000)(nltk.stem.WordNetLemmatizer().lemmatize)
        self.tokenize = nltk.tokenize.word_tokenize


    def __call__(self, text):
        tokens = self.tokenize(text)
        stopwords_list = nltk.corpus.stopwords.words('english')
        removed_tokens = ["br", "``", "--", "..."]
        tokens = [self.stem(token.lower()) for token in tokens if not (
                                                                token in stopwords_list
                                                                or token in removed_tokens
                                                                or token[0] == "'"
                                                                or (not token.isnumeric() and len(token)<=1)
                                                                       )]
        return tokens
