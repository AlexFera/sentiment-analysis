import argparse
import os
import pickle
import sys
from time import time

from sklearn.naive_bayes import BernoulliNB

import common


# As the data set for training we are using the Large Movie Review Data Set v1.0
# Link to the data set: http://ai.stanford.edu/~amaas/data/sentiment/


def create_arg_parser():
    """"Creates and returns the ArgumentParser object for our application"""

    parser = argparse.ArgumentParser(description="Script that uses the Naive Bayes Algorithm for sentiment analysis.")
    parser.add_argument("-pd", "--positive-data-directory",
                        help="Path to the positive data set directory used for training.",
                        default="../../aclImdb_v1/aclImdb/train/pos/")

    parser.add_argument("-nd", "--negative-data-directory",
                        help="Path to the negative data set directory used for training.",
                        default="../../aclImdb_v1/aclImdb/train/neg/")

    return parser


def get_reviews(directory_path):
    """"This function takes an absolute directory path, parses each file found and
    appends the content of the file to a list"""
    lines = []
    for root, dirs, files in os.walk(directory_path, onerror=common.onerror):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                f_content = f.read()
                lines.append(f_content)

    return lines


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
    vectorizer = common.StemmedCountVectorizer(min_df=3, binary="true", analyzer="word")
    t0 = time()
    document_term_matrix = vectorizer.fit_transform(documents)
    print("done in %0.3fs" % (time() - t0))

    return document_term_matrix, vectorizer


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    reviews = get_all_reviews(parsed_args.positive_data_directory, parsed_args.negative_data_directory)
    train_documents = [review[0] for review in reviews]
    train_labels = [int(review[1]) for review in reviews]
    term_frequency_matrix, count_vectorizer = get_term_frequency_matrix(train_documents)
    classifier = BernoulliNB().fit(term_frequency_matrix, train_labels)

    pickle.dump(classifier, open(common.classifier_path, "wb"))
    pickle.dump(count_vectorizer, open(common.count_vectorizer_path, "wb"))
