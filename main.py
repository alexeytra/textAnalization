import sklearn
import numpy
import voсabulary as voc
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sentaces_train, action_train, sentaces_test, action_test = voc.getCSVSample()

vocabulary = CountVectorizer()
vocabulary.fit(sentaces_train)
X_train = vocabulary.transform(sentaces_train)
vocabulary.fit(action_train)
y_train = vocabulary.transform(sentaces_train)

input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=100, verbose=False, batch_size=10)


vocabulary.fit(sentaces_train)
sentence = vocabulary.transform(voc.getCSVTest())
print(model.predict(sentence))
