from dataloader import load_data
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


sample_submission_data_path = '../data/sample_submission.csv'
train_data_path = '../data/train.csv'
test_data_path = '../data/test.csv'

def load_data(path):
    return pd.read_csv(path)

def train_svm():
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    X = train_data['text']
    y = train_data['sentiment']
    import pdb;pdb.set_trace()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1,test_size=0.1)






if __name__ == '__main__':
    train_svm()
