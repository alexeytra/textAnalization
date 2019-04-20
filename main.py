import sklearn
import numpy
import voсabulary as voc
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
import os
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sentences_train, action_train, sentences_test, action_test = voc.getCSVSample()


#
# vocabulary = CountVectorizer()
# vocabulary.fit(sentences_train)
# X_train = vocabulary.transform(sentences_train)
# print(X_train)
#
# input_dim = X_train.shape[1]
# model = Sequential()
# model.add(layers.Dense(13, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# history = model.fit(X_train, y_train, epochs=100, verbose=False, batch_size=5)
#
#
# vocabulary.fit(sentences_train)
# sentence = vocabulary.transform(voc.getCSVTest())
# print(model.predict_classes(sentence))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_matrix(sentences_train, mode='tfidf')
X_test = tokenizer.texts_to_matrix(sentences_test, mode='tfidf')

print(action_train)
encoder = LabelBinarizer()
encoder.fit(action_train)
y_train = encoder.transform(action_train)
y_test = encoder.transform(action_test)


model = Sequential()
model.add(layers.Dense(512, input_shape=(X_train.shape[1],)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3))
model.add(layers.Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=5, epochs=40, verbose=1, validation_split=0.1)

score = model.evaluate(X_test, y_test, batch_size=2, verbose=1)
print('Test accuracy:', score[1])