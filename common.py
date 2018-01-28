import argparse


def onerror(error):
    raise Exception(error)


def create_arg_parser():
    """"Creates and returns the ArgumentParser object for our application"""

    parser = argparse.ArgumentParser(description="Script that uses the Naive Bayes Algorithm for sentiment analysis.")
    parser.add_argument("-pd", "--positive-data-directory",
                        help="Path to the positive data set directory used for training.",
                        default="../../aclImdb_v1/aclImdb/train/pos/")

    parser.add_argument("-nd", "--negative-data-directory",
                        help="Path to the negative data set directory used for training.",
                        default="../../aclImdb_v1/aclImdb/train/neg/")

    parser.add_argument("-td", "--test-data-directory",
                        help="Path to the test data directory tha contains comments unclassified.",
                        default="../../test-folder/")

    parser.add_argument("-tp", "--test-phrase",
                        help="A str",
                        default="../../test-folder/")

    return parser
