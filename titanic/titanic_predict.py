from titanic_math import sigmoid,dot


def predict(X,y,theta):
    predictions = []
    correct = 0
    for i,x in enumerate(X):
        h = sigmoid(dot(x,theta))
        predictions.append(1 if h >= .5 else 0)
        if predictions[i] == y[i]: correct+=1

    return correct*100/len(y)
