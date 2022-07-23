import argparse


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--a', default=3, type=int)
args = parser.parse_args()