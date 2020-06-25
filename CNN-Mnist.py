import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import seaborn as sns
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0], cmap='Greys')

y_cat_train = to_categorical(y_train, num_classes=10)

y_cat_test = to_categorical(y_test, num_classes=10)

X_train = X_train / 250

X_test = X_test / 250

X_train = X_train.reshape(60000, 28, 28, 1)

X_test = X_test.reshape(10000, 28, 28, 1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("my_model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

score = model.evaluate(X_test, y_cat_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



