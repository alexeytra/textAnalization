import sklearn
import numpy as np
import voсabulary as voc
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
import os
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sentences_train, action_train, sentences_test, action_test = voc.getCSVSample()
num_samples = 11

print(sentences_test)

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
model.add(layers.Dense(num_samples, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=5, epochs=35, verbose=1, validation_split=0.1)

score = model.evaluate(X_test, y_test, batch_size=2, verbose=1)
print('Test accuracy:', score[1])

text_labels = encoder.classes_
prediction = model.predict(np.array([X_test[1]]))
print(prediction)
predicted_label = text_labels[np.argmax(prediction[0])]
print('Actual label: ' + action_test.iloc[0])
print("Predicted label: " + predicted_label)

