import argparse
import os
import pickle
import sys

import common


def create_arg_parser():
    """"Creates and returns the ArgumentParser object for our application"""

    parser = argparse.ArgumentParser(description="Script that loads the saved model and attempts predictions.")
    parser.add_argument("-tpd", "--test-positive-data-directory",
                        help="Path to the test data directory..",
                        default="../../aclImdb_v1/aclImdb/test/pos/")

    parser.add_argument("-tnd", "--test-negative-data-directory",
                        help="Path to the test data directory..",
                        default=".../../aclImdb_v1/aclImdb/train/neg/")

    parser.add_argument("-tp", "--test-phrase",
                        help="Provided test phrase to analyze sentiment.")

    return parser


def get_test_data(directory_path):
    lines = []
    abs_path = os.path.abspath(directory_path)
    for root, dirs, files in os.walk(abs_path, onerror=common.onerror):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                f_content = f.read()
                lines.append(f_content)

    return lines


def get_results(directory_path):
    pos_number = 0
    neg_number = 0
    test_data = get_test_data(directory_path)
    for line in test_data:
        review_to_predict = count_vectorizer.transform([line])
        result = classifier.predict(review_to_predict)
        if result[0] == 1:
            pos_number += 1
        else:
            neg_number += 1

    return pos_number, neg_number


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    classifier = pickle.load(open(common.classifier_path, "rb"))
    count_vectorizer = pickle.load(open(common.count_vectorizer_path, "rb"))

    if parsed_args.test_phrase is not None:
        to_predict = count_vectorizer.transform([parsed_args.test_phrase])
        prediction_result = classifier.predict(to_predict)
        if prediction_result[0] == 1:
            print("It's a positive statement!")
        else:
            print("It's a negative statement!")
    else:
        print("For the positive reviews test from the folder the results are:")
        number_pos, number_neg = get_results(parsed_args.test_positive_data_directory)
        print("Number of pos:", number_pos)
        print("Number of neg:", number_neg)

        print("For the negative reviews from the test folder the results are:")
        number_pos, number_neg = get_results(parsed_args.test_negative_data_directory)
        print("Number of pos:", number_pos)
        print("Number of neg:", number_neg)
