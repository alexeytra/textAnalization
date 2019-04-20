import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy

def getCSVSample():
    df_list = []
    table = pd.read_csv('sample.csv', names=['x', 'r', 'y', 'action'])
    df_list.append(table)
    df = pd.concat(df_list)
    df['sentence'] = df['x'] + ' ' + df['r'] + ' ' + df['y']
    sentences = df['sentence']
    action = df['action']

    sentences_train, sentences_test, action_train, action_test = train_test_split(sentences, action, test_size=0.10, random_state=1000)
    return sentences_train, action_train, sentences_test, action_test

def getCSVTest():
    df_list = []
    table = pd.read_csv('sample.csv', names=['x', 'r', 'y', 'action'])
    df_list.append(table)
    df = pd.concat(df_list)

    df['sentence'] = df['x'] + ' ' + df['r'] + ' ' + df['y']
    return df['sentence']


