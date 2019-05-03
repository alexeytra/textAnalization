import matplotlib.pyplot as plt
import voсabulary as voc
import pandas as pd
from keras.preprocessing import sequence

plt.style.use('ggplot')
import numpy as np


class Service:

    def __init__(self, encoder, tokenizer, model):
        self.__encoder = encoder
        self.__tokenizer = tokenizer
        self.__model = model
        self.__test = voc.getTest()

    def plot_history(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Точность обучения')
        plt.plot(x, val_acc, 'r', label='Точность проверки')
        plt.title('Обучение и проверка (Точность)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Функция поторь обучения')
        plt.plot(x, val_loss, 'r', label='Функция потерь проверки')
        plt.title('Обучение и проверка (loss function)')
        plt.legend()
        plt.show()

    def predictio_nmlp(self):
        y_test = self.__tokenizer.texts_to_matrix(self.__test, mode='tfidf')
        self.__prediction(y_test)

    def prediction_cnn(self, max_len):
        cnn_texts_seq = self.__tokenizer.texts_to_sequences(self.__test)
        y_test = sequence.pad_sequences(cnn_texts_seq, maxlen=max_len)
        self.__prediction(y_test)

    def __prediction(self, y_test):
        text_labels = self.__encoder.classes_
        prediction = self.__model.predict(np.array([y_test[1]]))
        predicted_label = text_labels[np.argmax(prediction[0])]
        print("Sentence: " + self.__test[1])
        print("Predicted label: " + predicted_label)
