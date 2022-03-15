import numpy as np
import sklearn.neighbors
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(self, k=5, strategy="brute", metric="euclidean", weights=False, test_block_size=None):
        self.k = k
        self.strategy = strategy
        self.str_metric = metric
        if metric == "euclidean":
            self.metric = euclidean_distance
        elif metric == "cosine":
            self.metric = cosine_distance
        self.weights = weights
        self.test_block_size = test_block_size
        self.model = None
        self.classes = None
        self.find_in_blocks = self.find_in_blocks
        self.y = None
        self.X = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).astype(np.int)
        self.classes = np.max(self.y) + 1
        if self.strategy != "my_own":
            self.model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.k,
                algorithm=self.strategy,
                metric=self.str_metric
            )
            self.model.fit(X)

    def find_kneighbors(self, X, return_distance=False):
        data = np.array(X)
        if return_distance:
            dist_matr = np.zeros((data.shape[0], self.k))
        else:
            dist_matr = None
        neighbors = np.zeros((data.shape[0], self.k))
        if self.test_block_size is None:
            block_size = data.shape[0]
        else:
            block_size = min(self.test_block_size, data.shape[0])
        N = data.shape[0] // block_size
        r = data.shape[0] % block_size
        if r != 0:
            N += 1
        left = 0
        if N == 1 and r != 0:
            right = r
        else:
            right = block_size
        for n in range(N):
            if return_distance:
                if self.strategy != "my_own":
                    dist_matr[left:right], neighbors[left:right] = \
                        self.model.kneighbors(data[left:right], self.k, return_distance)
                else:
                    dist_matr[left:right], neighbors[left:right] = \
                        self.find_in_blocks(data[left:right], return_distance)
            else:
                if self.strategy != "my_own":
                    neighbors[left:right] = self.model.kneighbors(data[left:right], self.k, return_distance)
                else:
                    neighbors[left:right] = self.find_in_blocks(data[left:right], return_distance)
            left += block_size
            if n == N - 2 and r != 0:
                right += r
            else:
                right += block_size
        if return_distance:
            return (dist_matr, neighbors)
        else:
            return neighbors

    def find_in_blocks(self, X, return_distance=False):
        distance = self.metric(X, self.X)
        neighbors = np.argsort(distance, axis=1)[:, 0:self.k]
        if return_distance:
            return (np.sort(distance, axis=1)[:, 0:self.k], neighbors)
        return neighbors

    def predict(self, X):
        votes = np.zeros((X.shape[0], self.classes))
        if self.weights:
            dist_matr, neighbors = self.find_kneighbors(X, True)
            eps = 0.00001
            for i in range(neighbors.shape[0]):
                for j in range(neighbors[i].shape[0]):
                    cl = int(neighbors[i][j])
                    ind = int(self.y[cl])
                    votes[i][ind] += 1 / (dist_matr[i][j] + eps)
            y_predict = np.argmax(votes, axis=1)
            return y_predict
        else:
            neighbors = self.find_kneighbors(X, False)
            dist_matr = None
            for i in range(neighbors.shape[0]):
                for j in range(neighbors[i].shape[0]):
                    cl = int(neighbors[i][j])
                    ind = int(self.y[cl])
                    votes[i][ind] += 1
            y_predict = np.argmax(votes, axis=1)
            return y_predict
