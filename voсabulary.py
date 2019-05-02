import pandas as pd
from sklearn.model_selection import train_test_split


def getSample():
    df_list = []
    table = pd.read_json('samples/train.json', encoding='utf-8', orient='index')
    df_list.append(table)
    df = pd.concat(df_list)
    df['sentence'] = df['x'] + ' ' + df['R'] + ' ' + df['y']
    sentences = df['sentence']
    actions = df['action']

    sentences_train, sentences_test, action_train, action_test = train_test_split(sentences, actions, test_size=0.01,
                                                                                  random_state=1000)
    return sentences_train, action_train, sentences_test, action_test, actions


def getTest():
    df_list = []
    table = pd.read_json('samples/test.json', encoding='utf-8', orient='index')
    df_list.append(table)
    df = pd.concat(df_list)

    df['sentence'] = df['x'] + ' ' + df['R'] + ' ' + df['y']
    return df['sentence']
