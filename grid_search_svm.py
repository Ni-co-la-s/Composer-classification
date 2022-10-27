import numpy as np
import os
import random
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from utils import N_CHROMA, N_MFCC, L_composer
from sklearn.metrics import classification_report


# Retrieve the pickle dataset and find the best parameters for PCA and SVM
# via a grid search (5-fold validation)
XX = []
labels = []
d = {}
nb_dimension = N_CHROMA + N_MFCC


k = 0
filename = "Dataset_train_test" + str(k) + ".pkl"

while os.path.exists(filename):
    file_to_read = open(filename, "rb")

    loaded_object = pickle.load(file_to_read)
    for i in range(len(loaded_object[0])):
        X_int = []
        for j in loaded_object[0][i]:
            X_int.append(j)
        if len(X_int) == nb_dimension and loaded_object[1][i] in L_composer:
            XX.append(X_int)

            labels.append(loaded_object[1][i])
    k = k + 1
    filename = "Dataset_train_test" + str(k) + ".pkl"
# Randomize the vectors retrieves for the training
combined = list(zip(XX, labels))
random.shuffle(combined)
X, y = [], []

XX[:], labels[:] = zip(*combined)

for i in range(len(labels)):
    if labels[i] not in d:
        d[labels[i]] = 1
    else:
        d[labels[i]] += 1
    # Limits the number of vectors retrieved
    if d[labels[i]] < 1000:
        X.append(XX[i])
        y.append(labels[i])
XX_validation = []
labels_validation = []
d_val = {}

k = 0
filename = "Dataset_validation" + str(k) + ".pkl"

while os.path.exists(filename):
    file_to_read = open(filename, "rb")

    loaded_object = pickle.load(file_to_read)
    for i in range(len(loaded_object[0])):
        X_int = []
        for j in loaded_object[0][i]:
            X_int.append(j)
        if len(X_int) == nb_dimension and loaded_object[1][i] in L_composer:
            XX_validation.append(X_int)

            labels_validation.append(loaded_object[1][i])
    k = k + 1
    filename = "Dataset_train_test" + str(k) + ".pkl"
combined = list(zip(XX_validation, labels_validation))
random.shuffle(combined)
X_val, y_val = [], []
X_val[:], y_val[:] = zip(*combined)


for i in y_val:
    if i not in d_val:
        d_val[i] = 1
    else:
        d_val[i] += 1
print("Components of training dataset")
print(d)
print("Components of validation dataset")
print(d_val)


X = np.array(X)
X_val = np.array(X_val)
y = np.array(y)
y_val = np.array(y_val)


nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx * ny))

nsamples_test, nx, ny = X_val.shape
X_val = X_val.reshape((nsamples_test, nx * ny))

print("scaling")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

print("grid_search")


svm = SVC()
pca = PCA()

pipe = Pipeline(steps=[("pca", pca), ("svm", svm)])

params_grid = {
    "svm__C": [0.001, 0.01, 0.1, 1],
    "svm__kernel": ["linear"],
    "svm__gamma": [0.001, 0.01, 0.1, 1],
    "pca__n_components": [10, 50, 100],
}

grid = GridSearchCV(pipe, params_grid, refit=True, verbose=3)
grid.fit(X, y)

print(grid.best_params_)


print(grid.best_estimator_)


grid_predictions = grid.predict(X_val)

print(classification_report(y_val, grid_predictions))
print(i, confusion_matrix(y_val, grid_predictions))
