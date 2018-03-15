import titanic_math as tmath
from math import log


def add_bias(X): return [[1]+x for x in X]


def logistic_cost(X, y, theta, lamb):
    m = len(y)
    n = len(theta)

