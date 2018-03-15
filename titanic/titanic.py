import titanic_examples as tex
import titanic_math as tmath
import titanic_regression as treg
import titanic_predict as tpred


def init(filename):
    # Learning rate and regularization coefficients
    lamb = 0
    alpha = 1

    # Read in data and initialize theta vector
    data = tex.parse_examples(filename)
    X = treg.add_bias(data[0])
    y = data[1]
    theta = tmath.rmatrix(1,len(X[0]))[0]

    # Train algorithm
    theta = treg.logistic_reg(X,y,theta,lamb,alpha)

    # Check training set fit
    print('Training accuracy: {}'.format(tpred.predict(X,y,theta)))
