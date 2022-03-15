import random
from nearest_neighbors import KNNClassifier
from copy import copy
import numpy as np


def kfold(n, n_folds):
    lst = list(range(n))
    res = []
    size = n // n_folds
    r = n % n_folds
    left = 0
    right = None
    for i in range(n_folds):
        if i < r:
            cur_size = size + 1
        else:
            cur_size = size
        right = left + cur_size
        y_test = np.array(lst[left:right]).astype(int)
        if left == 0:
            y_train = np.array(lst[right:]).astype(int)
        elif right == 0:
            y_train = np.array(lst[:left]).astype(int)
        else:
            y_train = np.hstack((np.array(lst[:left]), np.array(lst[right:]))).astype(int)
        res.append((y_train, y_test))
        left += cur_size
    return res


def new_predict(model, neighbors, dist_matr=None):
    votes = np.zeros((neighbors.shape[0], np.max(model.y) + 1))
    if model.weights:
        eps = 0.00001
        for i in range(neighbors.shape[0]):
            for j in range(neighbors[i].shape[0]):
                cl = int(neighbors[i][j])
                ind = int(model.y[cl])
                votes[i][ind] += 1 / (dist_matr[i][j] + eps)
        y_predict = np.argmax(votes, axis=1)
        return y_predict
    else:
        for i in range(neighbors.shape[0]):
            for j in range(neighbors[i].shape[0]):
                cl = int(neighbors[i][j])
                ind = int(model.y[cl])
                votes[i][ind] += 1
        y_predict = np.argmax(votes, axis=1)
        return y_predict


def accuracy(y_true, y_pred):
    length = len(y_true)
    true_classes = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true_classes += 1
    return true_classes / length


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    res = dict()
    if cv is None:
        folds = kfold(X.shape[0], 3)
    else:
        folds = cv
    model = KNNClassifier(k=k_list[-1], **kwargs)
    for s in folds:
        model.fit(X[s[0]], y[s[0]])
        dist = None
        if not model.weights:
            neighbors = model.find_kneighbors(X[s[1]], return_distance=False)
        else:
            (dist, neighbors) = model.find_kneighbors(X[s[1]], return_distance=True)
        for k in k_list[::-1]:
            if dist is not None:
                dist = dist[:, :k]
            neighbors = neighbors[:, :k]
            pred = new_predict(model, neighbors, dist)
            acc = accuracy(y[s[1]], pred)
            if k in res:
                res[k] = np.append(res[k], acc)
            else:
                res[k] = np.array([acc])
    return res
