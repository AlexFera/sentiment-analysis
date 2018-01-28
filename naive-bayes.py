import os
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

import common


def get_reviews(directory_path):
    """"This function takes an absolute directory path, parses each file found and
    appends the content of the file to a list"""
    reviews = []
    for root, dirs, files in os.walk(directory_path, onerror=common.onerror):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                f_content = f.read()
                reviews.append(f_content)

    return reviews


def get_all_reviews():
    """This function gets all the positive and negative reviews from the training folder
    and returns all list of list, where the first element is the text of the review and
    the second element is the label, 1 for a positive review and 0 for a negative review"""
    pos_train_path = os.path.abspath('../../aclImdb_v1/aclImdb/train/pos/')
    neg_train_path = os.path.abspath('../../aclImdb_v1/aclImdb/train/neg/')
    pos_reviews = get_reviews(pos_train_path)
    neg_reviews = get_reviews(neg_train_path)

    all_reviews = []
    for pos_review in pos_reviews:
        review = [pos_review, '1']
        all_reviews.append(review)

    for neg_review in neg_reviews:
        review = [neg_review, '0']
        all_reviews.append(review)

    return all_reviews


def get_term_frequency_matrix(documents):
    """This function transforms the text training data representation
    into numerical features; In this case into term frequency representation.
    The data is represented as matrix of token counts."""
    print("Extracting term frequency...")
    vectorizer = CountVectorizer(binary="true")
    t0 = time()
    document_term_matrix = vectorizer.fit_transform(documents)
    print("done in %0.3fs" % (time() - t0))

    return document_term_matrix, vectorizer


def train():
    reviews = get_all_reviews()
    train_documents = [review[0] for review in reviews]
    term_frequency_matrix, vectorizer = get_term_frequency_matrix(train_documents)
    train_labels = [int(review[1]) for review in reviews]

    bernoulli_classifier = BernoulliNB().fit(term_frequency_matrix, train_labels)

    return bernoulli_classifier, vectorizer


classifier, count_vectorizer = train()

label = classifier.predict(count_vectorizer.transform([
    "The third movie produced by Howard Hughes, this gem was thought to be lost. It was recently restored and shown on TCM (12/15/04). The plot is a familiar one - two WW I soldiers escape from a German prison camp (guarded by an extremely lethargic German shepherd, who practically guides them out of the camp), stow away on a ship, and end up in Arabia, where they rescue the lovely Mary Astor. The restoration is very good overall, although there are two or three very rough sequences. The production is very good, and there are some very funny scenes. And did I mention that Mary Astor is in it? The film won an Academy Award for the now-defunct category of Best Direction of a Comedy."]))
label2 = classifier.predict(count_vectorizer.transform(["This is the worst movie"]))

print(label[0])
print(label2[0])

print("The end")
