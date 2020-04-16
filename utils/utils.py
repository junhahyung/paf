import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='source_estimator_config.json',
        help='The Configuration file')
    print(arg)
    args = argparser.parse_args()
    return args
