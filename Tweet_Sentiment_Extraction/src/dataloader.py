import csv
import pandas as pd
import numpy as np

sample_submission_data_path = '../data/sample_submission.csv'
train_data_path = '../data/train.csv'
test_data_path = '../data/test.csv'

def load_data():
    with open(sample_submission_data_path, 'r') as f:
        reader = csv.reader(f)
        sample_submission_data = [row for row in reader]

    sample_submission_df = pd.read_csv(sample_submission_data_path)
    train_data_df = pd.read_csv(train_data_path)
    test_data_df = pd.read_csv(test_data_path)

    return sample_submission_data, sample_submission_df, train_data_df, test_data_df

if __name__ == '__main__':
    sample_submission_data_path, sample_submission_df, train_data_df, test_data_df = load_data()
    print(train_data_df)