from math import e
import random


def transpose(X):
    assert len(X) and len(X[0])
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]


def rmatrix(nrow, ncol): return [[random.uniform(0,1) for i in range(ncol)] for j in range(nrow)]


def dot(u, v):
    assert len(u) == len(v)
    return sum([u[i]*v[i] for i in range(len(u))])


def sigmoid(z):
    try: h = 1 / (1 + e**-z)
    except: h = 0.0
    return h
