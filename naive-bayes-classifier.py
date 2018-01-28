import os
import pickle
import sys
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


def get_all_reviews(pos_directory, neg_directory):
    """This function gets all the positive and negative reviews from the training folder
    and returns all list of list, where the first element is the text of the review and
    the second element is the label, 1 for a positive review and 0 for a negative review"""
    pos_train_path = os.path.abspath(pos_directory)
    neg_train_path = os.path.abspath(neg_directory)
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


def train(reviews):
    """Learn a Naive Bayes classifier on the transformed data"""
    train_documents = [review[0] for review in reviews]
    term_frequency_matrix, vectorizer = get_term_frequency_matrix(train_documents)
    train_labels = [int(review[1]) for review in reviews]

    bernoulli_classifier = BernoulliNB().fit(term_frequency_matrix, train_labels)

    return bernoulli_classifier, vectorizer


if __name__ == "__main__":
    arg_parser = common.create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    reviews = get_all_reviews(parsed_args.positive_data_directory, parsed_args.negative_data_directory)
    # classifier, count_vectorizer = train(reviews)

    classifier_path = "naive_bayes_model.bin"
    count_vectorizer_path = "count_vectorizer.bin"
    # pickle.dump(classifier, open(classifier_path, "wb"))
    # pickle.dump(count_vectorizer, open(count_vectorizer_path, "wb"))
    classifier = pickle.load(open(classifier_path, "rb"))
    count_vectorizer = pickle.load(open(count_vectorizer_path, "rb"))

    if parsed_args.test_phrase is not None:
        to_predict = count_vectorizer.transform([parsed_args.test_phrase])
        prediction_result = classifier.predict(to_predict)
        if prediction_result[0] == 1:
            print("It's a positive statement!")
        else:
            print("It's a negative statement!")
