from keras import Sequential
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
import voсabulary as voc
import pandas as pd

from neural_networks.service import Service


class MLPModel:
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
        self.__num_samples = pd.Series(self.__actions, name='A').unique()

        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__sentences_train)

        self.__X_train = self.__tokenizer.texts_to_matrix(self.__sentences_train, mode='tfidf')
        self.__X_test = self.__tokenizer.texts_to_matrix(self.__sentences_test, mode='tfidf')

        self.__encoder = LabelBinarizer()
        self.__encoder.fit(self.__action_train)
        self.__y_train = self.__encoder.transform(self.__action_train)
        self.__y_test = self.__encoder.transform(self.__action_test)

    def activate_mlp_model_v1(self):
        self.__text_preproccessing()

        model = Sequential()
        model.add(layers.Dense(200, input_shape=(self.__X_train.shape[1],), activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.__num_samples.size, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(self.__X_train, self.__y_train, batch_size=5, epochs=35, verbose=1, validation_split=0.1)
        # score = model.evaluate(X_test, y_test, batch_size=2, verbose=1)

        serviceNNs = Service(self.__encoder, self.__tokenizer, model)
        serviceNNs.plot_history(history)
        serviceNNs.prediction_mlp()



