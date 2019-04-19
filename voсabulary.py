import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def getCSVSample():
    df_list = []
    table = pd.read_csv('sample.csv', names=['x', 'r', 'y', 'action'])
    df_list.append(table)
    df = pd.concat(df_list)

    sentences = df['x'].values + " " + df['r'].values + " " + df['y'].values
    action = df['action'].values

    sentences_train, sentences_test, action_train, action_test = train_test_split(sentences, action, test_size=0.5, random_state=1000)
    return sentences_train, action_train, sentences_test, action_test
    # vocabulary = CountVectorizer(min_df=0, lowercase=False)
    # vocabulary.fit(sentences)

def getCSVTest():
    df_list = []
    table = pd.read_csv('sample.csv', names=['x', 'r', 'y', 'action'])
    df_list.append(table)
    df = pd.concat(df_list)

    sentences = df['x'].values + " " + df['r'].values + " " + df['y'].values
    return sentences


