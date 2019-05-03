from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import time
from keras import metrics
import pandas as pd
import voсabulary as voc
from neural_networks.service import Service


class CNNModel:

    def __init__(self):
        self.__sentences_train = None
        self.__action_train = None
        self.__sentences_test = None
        self.__action_test = None
        self.__actions = None
        self.__max_len = None
        self.__X_train = None
        self.__y_train = None
        self.__X_test = None
        self.__y_test = None
        self.__num_actions = None
        self.__tokenizer = None
        self.__encoder = None


    def __text_preproccessing(self):
        self.__sentences_train, self.__action_train, self.__sentences_test, self.__action_test, self.__actions = voc.getSample()
        self.__num_actions = pd.Series(self.__actions, name='A').unique()
        self.__max_len = 100

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.__sentences_train)
        cnn_texts_seq = tokenizer.texts_to_sequences(self.__sentences_train)
        self.__X_train = sequence.pad_sequences(cnn_texts_seq, maxlen=self.__max_len)

        encoder = LabelBinarizer()
        encoder.fit(self.__action_train)
        self.__y_train = encoder.transform(self.__action_train)
        self.__y_test = encoder.transform(self.__action_test)

        self.__tokenizer = tokenizer
        self.__encoder = encoder

    def get_cnn_model_v1(self):
        self.__text_preproccessing()

        model = Sequential()
        model.add(Embedding(1000, 20, input_length=self.__max_len))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        history = model.fit(self.__X_train, self.__y_train, batch_size=5, epochs=30, verbose=1, validation_split=0.1)

        serviceNNs = Service(self.__encoder, self.__tokenizer, model)
        serviceNNs.plot_history(history)
        serviceNNs.prediction_cnn(self.__max_len)
