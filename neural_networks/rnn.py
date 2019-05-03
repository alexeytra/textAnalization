from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pandas as pd
import voсabulary as voc
from neural_networks.service import Service


class RNNModel:
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
        self.__num_samples = None

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
        self.__num_samples = pd.Series(self.__actions, name='A').unique()


    def activate_rnn_model_v1(self):
        self.__text_preproccessing()
        # create the model
        model = Sequential()
        model.add(Embedding(1000, 32, input_length=self.__max_len))
        model.add(LSTM(100))
        model.add(Dense(self.__num_samples.size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        history = model.fit(self.__X_train, self.__y_train, epochs=20, batch_size=5, verbose=1, validation_split=0.1)

        service = Service(self.__encoder, self.__tokenizer, model)
        service.plot_history(history)
        service.prediction_cnn(self.__max_len)

    def activate_rnn_model_v2(self):
        self.__text_preproccessing()
        # create the model
        model = Sequential()
        model.add(Embedding(1000, 32, input_length=self.__max_len))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(self.__num_samples.size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        history = model.fit(self.__X_train, self.__y_train, epochs=20, batch_size=5, verbose=1, validation_split=0.1)

        service = Service(self.__encoder, self.__tokenizer, model)
        service.plot_history(history)
        service.prediction_cnn(self.__max_len)
