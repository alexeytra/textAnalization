from keras import Sequential
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import voсabulary as voc
import pandas as pd

plt.style.use('ggplot')


def call_mlpNN():
    sentences_train, action_train, sentences_test, action_test, actions = voc.getSample()
    num_samples = pd.Series(actions, name='A').unique()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_matrix(sentences_train, mode='tfidf')
    X_test = tokenizer.texts_to_matrix(sentences_test, mode='tfidf')


    encoder = LabelBinarizer()
    encoder.fit(action_train)
    y_train = encoder.transform(action_train)
    y_test = encoder.transform(action_test)

    model = Sequential()
    model.add(layers.Dense(200, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_samples.size, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=5, epochs=35, verbose=1, validation_split=0.1)
    plot_history(history)
    score = model.evaluate(X_test, y_test, batch_size=2, verbose=1)

    prediction(tokenizer, encoder, model)



def plot_history(history):
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


def prediction(tokenizer, encoder, model):

    test = voc.getTest()
    y_test = tokenizer.texts_to_matrix(test, mode='tfidf')
    text_labels = encoder.classes_
    prediction = model.predict(np.array([y_test[1]]))

    predicted_label = text_labels[np.argmax(prediction[0])]
    print("Sentence: " + test[1])
    print("Predicted label: " + predicted_label)
