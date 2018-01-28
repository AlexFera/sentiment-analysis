import argparse
import pickle
import sys

import common


def create_arg_parser():
    """"Creates and returns the ArgumentParser object for our application"""

    parser = argparse.ArgumentParser(description="Script that loads the saved model and attempts predictions.")
    parser.add_argument("-td", "--test-data-directory",
                        help="Path to the test data directory..",
                        default="../../test-data")

    parser.add_argument("-tp", "--test-phrase",
                        help="Provided test phrase to analyze sentiment.", required=True)

    return parser


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
