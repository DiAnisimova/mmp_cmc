import numpy as np
import scipy
from scipy import special


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        M = y * (X @ w)
        res = np.sum(np.logaddexp(0, -M)) / X.shape[0] + self.coef * (w.dot(w.T)) / 2
        return res

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        if isinstance(X, np.ndarray):
            M = -y * (X.dot(w))
            denom = scipy.special.expit(M).reshape(-1, 1)
            res = -np.sum(X * y.reshape(-1, 1) * denom, axis=0).reshape(-1) / X.shape[0]
            res += self.coef * w
            return res
        if isinstance(X, scipy.sparse.csr_matrix):
            M = -y * (X.dot(w))
            denom = scipy.special.expit(M).reshape(-1,1)
            res = -np.sum(X.multiply(y.reshape(-1, 1)).multiply(denom),axis=0) / X.shape[0]
            res += self.coef * w
            return np.ravel(res)
