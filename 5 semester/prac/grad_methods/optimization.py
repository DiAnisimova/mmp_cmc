import oracles
import time
from copy import copy
import scipy
import numpy as np


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_func = loss_function
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y, X_test=None, y_test=None, w_0=None, trace=False, accur=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        history = {'time': [0.0], 'func': [], 'help_time': [0.0]} 
        if accur:
          history['accuracy'] = [0.0]
        if w_0 is None:
            cur_w0 = np.zeros(X.shape[1])
        else:
            cur_w0 = w_0
        if self.loss_func == 'binary_logistic':
            self.oracle = oracles.BinaryLogistic(**self.kwargs)
        history['func'].append(self.oracle.func(X, y, cur_w0))

        prev_time = time.time()
        prev_loss = history['func'][0]
        cur_w = cur_w0

        for i in range(self.max_iter):
            new_time = time.time()

            eta = self.alpha / ((i + 1) ** self.beta)
            cur_w = cur_w - (eta * self.oracle.grad(X, y, cur_w))
            self.w = copy(cur_w)
            cur_loss = self.oracle.func(X, y, cur_w)
            if accur:
                y_pred = self.predict(X_test)
                acc = accuracy(y_pred, y_test)
                history['accuracy'].append(acc)
            history['time'].append(new_time - prev_time)
            history['help_time'].append(history['help_time'][-1] + history['time'][-1])
            history['func'].append(cur_loss)
            
            if abs(prev_loss - cur_loss) < self.tolerance:
                break
            prev_time = copy(new_time)
            prev_loss = copy(cur_loss)
        self.w = cur_w
        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X.dot(self.w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        prob1 = scipy.special.expit(X.dot(self.w))
        prob2 = 1 - prob1
        return np.array((prob1.T, prob2.T)).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_func = loss_function
        self.batch_size = batch_size
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, accur=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        history = {'epoch_num': [0.0], 'time': [0.0], 'func': [], 'weights_diff': [0.0], 'help_time': [0.0]}
        if accur:
            history['accuracy'] = [0.0]

        if w_0 is None:
            cur_w0 = np.zeros(X.shape[1])
        else:
            cur_w0 = w_0

        if self.loss_func == 'binary_logistic':
            self.oracle = oracles.BinaryLogistic(self.kwargs['l2_coef'])
        history['func'].append(self.oracle.func(X, y, cur_w0))

        prev_time = time.time()
        prev_loss = history['func'][0]
        w_in_dict = copy(cur_w0)

        cur_loss = None
        cur_epoch = None
        cur_w = copy(cur_w0)

        objects = 0
        flag = False

        for i in range(self.max_iter):
            self.indexes = np.arange(X.shape[0])
            np.random.shuffle(self.indexes)
            for j in range(0, X.shape[0], self.batch_size):
                new_time = time.time()
                eta = self.alpha / ((i + 1) ** self.beta)
                if j + self.batch_size < X.shape[0]:
                    cur_indexes = self.indexes[j:j + self.batch_size]
                    objects += self.batch_size
                else:
                    cur_indexes = self.indexes[j:X.shape[0]]
                    objects += X.shape[0] - j

                cur_epoch = objects / X.shape[0]
                cur_w -= (eta * self.oracle.grad(X[cur_indexes], y[cur_indexes], cur_w))
                self.w = copy(cur_w)
                cur_loss = self.oracle.func(X[cur_indexes], y[cur_indexes], cur_w)
                if abs(cur_epoch - history['epoch_num'][-1]) > log_freq:
                    history['time'].append(new_time - prev_time)
                    history['help_time'].append(history['help_time'][-1] + history['time'][-1])
                    history['epoch_num'].append(cur_epoch)
                    history['weights_diff'].append((cur_w - w_in_dict) @ (cur_w - w_in_dict).T)
                    history['func'].append(cur_loss)
                    if accur:
                        y_pred = self.predict(X_test)
                        acc = accuracy(y_pred, y_test)
                        history['accuracy'].append(acc)
                    w_in_dict = copy(cur_w)

                if abs(prev_loss - cur_loss) < self.tolerance:
                    break
                    flag = True
                prev_time = copy(new_time)
                prev_loss = copy(cur_loss)
            if flag:
                break

        self.w = cur_w
        if trace:
            return history

