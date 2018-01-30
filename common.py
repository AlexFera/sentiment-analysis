import nltk.stem

from sklearn.feature_extraction.text import CountVectorizer


def onerror(error):
    raise Exception(error)


classifier_path = "naive_bayes_model.bin"
count_vectorizer_path = "count_vectorizer.bin"


# Extend the standard CountVectorizer with stemming ability
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer("english")
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])
