import numpy as np
import pandas as pd
import cv2 as cv

import sklearn
from sklearn.neighbors import KNeighborsClassifier

import pickle

### This file was used to train the KNN model and is only kept in case the model needs to be retrained for whatever reason

data = pd.read_csv('color_data_lab.csv', names=['L', 'a', 'b', 'color'])
data = data.iloc[1:, :]

data.drop_duplicates(inplace=True, ignore_index=True)

x = np.array(data.drop(['color'], axis = 1))
y = np.array(data['color'])

y_encoded = []
for elem in y:
    if elem == 'white':
        y_encoded.append(0)
    elif elem == 'orange':
       y_encoded.append(1)
    elif elem == 'green':
        y_encoded.append(2)
    elif elem == 'red':
        y_encoded.append(3)
    elif elem == 'blue':
        y_encoded.append(4)
    elif elem == 'yellow':
        y_encoded.append(5)

y_encoded = np.array(y_encoded)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y_encoded, test_size=0.1)

model = KNeighborsClassifier(n_neighbors = 5)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("Accuracy: ", accuracy)

filename = 'KNN_model_retrained'
pickle.dump(model, open(filename, 'wb'))
