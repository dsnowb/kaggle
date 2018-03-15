from math import e
import random


def transpose(X):
    assert len(X) and len(X[0])
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]


def rmatrix(nrow,ncol): return [[random.uniform(0,1) for i in range(ncol)] for j in range(nrow)]


def sigmoid(z): return 1 / (1 + e**-z)
