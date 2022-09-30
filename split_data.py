"""
This script will split data into training, validation and test sets
Args: train_percentage, valid_percentage, test_percentage, path
"""
import pandas as pd
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("--train_percentage", type=float, default=0.8)
args.add_argument("--valid_percentage", type=float, default=0.1)
args.add_argument("--test_percentage", type=float, default=0.1)
args.add_argument("--path", type=str, default="us-patent-phras-to-phrase-similarity/train.csv")


def split_data(train_percentage, valid_percentage, test_percentage, path):
    # Read in data
    df = pd.read_csv(path)
    # Split data into training, validation and test sets
    train = df.sample(frac=train_percentage, random_state=0)
    test = df.drop(train.index)
    valid = test.sample(frac=valid_percentage/(valid_percentage+test_percentage), random_state=0)
    test = test.drop(valid.index)
    # Write data to csv files
    train.to_csv("Patent_Phrase_Project/processed/train.csv", index=False)
    valid.to_csv("Patent_Phrase_Project/processed/valid.csv", index=False)
    test.to_csv("Patent_Phrase_Project/processed/test.csv", index=False)

def main():

    args = argparse.ArgumentParser()
    args.add_argument("--train_percentage", type=float, default=0.8)
    args.add_argument("--valid_percentage", type=float, default=0.1)
    args.add_argument("--test_percentage", type=float, default=0.1)
    args.add_argument("--path", type=str, default="/Users/alexandermoini/Downloads/us-patent-phrase-to-phrase-matching/train.csv")
    args = args.parse_args()
    split_data(args.train_percentage, args.valid_percentage, args.test_percentage, args.path)

if __name__ == "__main__":
    main()

