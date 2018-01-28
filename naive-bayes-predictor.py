import argparse
import os
import pickle
import sys

import common


def create_arg_parser():
    """"Creates and returns the ArgumentParser object for our application"""

    parser = argparse.ArgumentParser(description="Script that loads the saved model and attempts predictions.")
    parser.add_argument("-td", "--test-data-directory",
                        help="Path to the test data directory..",
                        default="../../test-folder")

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
        test_data = get_test_data(parsed_args.test_data_directory)
        for line in test_data:
            review_to_predict = count_vectorizer.transform([line])
            result = classifier.predict(review_to_predict)
            if result[0] == 1:
                print(line, " is a positive review!")
            else:
                print(line, " is a negative review")
