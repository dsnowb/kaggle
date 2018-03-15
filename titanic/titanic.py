import titanic_examples as tex
import titanic_math as tmath
import titanic_regression as treg


def init(filename):
    data = tex.parse_examples(filename)
    X = treg.add_bias(data[0])
    y = data[1]
    theta = tmath.rmatrix(1,len(X[0]))
    lamb = 0
    alpha = .01

    treg.logistic_reg(X,y,theta[0],lamb,alpha)
#    print(len(X))
#    print(len(y))
#    print(len(X[0]))
#    print(len(theta[0]))
#    print('X:\n{}'.format(X))
#    print('y:\n{}'.format(y))
#    print('lamb:\n{}'.format(lamb))
#    print('theta:\n{}'.format(theta))
