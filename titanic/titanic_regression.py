from titanic_math import dot, sigmoid
from math import log


def add_bias(X): return [[1]+x for x in X]


def logistic_cost(X, y, theta, lamb):
    assert len(X) == len(y) and len(X[0]) == len(theta)
    m = len(y)
    cost_sum = 0

    for i, x in enumerate(X):
        h = sigmoid(dot(x, theta))
        try: cost_sum = y[i]*log(h) + (1 - y[i])*log(1 - h)
        except: cost_sum = 0

    cost = -cost_sum / m + lamb*sum([theta[i]**2 for i in range(len(theta[1:]),1)]) / (2*m)
    return cost

def logistic_reg(X, y, theta, lamb, alpha):
    m = len(y)
    count = 0
    max_count = 1000
    cost = logistic_cost(X,y,theta,lamb)
    while cost > 0 and count < max_count:
        # update bias weight - no regularization
        new_theta = []
        grad_sum = 0
        for j, x in enumerate(X): grad_sum += (sigmoid(dot(x, theta)) - y[j])*x[0]
        grad = grad_sum / m
        new_theta.append(theta[0]-alpha*grad)

        # update other weights
        for i, weight in enumerate(theta[1:]):
            grad_sum = 0
            for j, x in enumerate(X): grad_sum += (sigmoid(dot(x, theta)) - y[j])*x[i]
            grad = (grad_sum + lamb*weight) / m
            new_theta.append(weight-alpha*grad)

        # check for divergence
        if not count%10:
            new_cost = logistic_cost(X,y,new_theta,lamb)
            if new_cost > cost:
                print('Gradient descent is diverging')
                break;
        
        # update values
        cost = new_cost
        theta = new_theta
        count += 1

    print('Final Cost: {}'.format(logistic_cost(X,y,theta,lamb)))

    return theta
