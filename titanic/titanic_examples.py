from math import sqrt
from titanic_math import transpose


def standardize(X):
    """
    Standardizes values in X by subtracting the feature mean element-wise, then dividing by the
    feature standard deviation. Where values for features are missing, 0 is substituted after
    mean normalization and feature scaling
    """
    Xt = transpose(X)
    n = len(Xt[0])
    for i in range(len(Xt)):
        num_known_vals = 0
        val_total = 0
        square_diff_sum = 0
        for j in range(n):
            if Xt[i][j] != '':
                val_total += Xt[i][j]
                num_known_vals += 1
                square_diff_sum += abs(Xt[i][j])**2
        mu = float(val_total)/num_known_vals
        sigma = sqrt(square_diff_sum/num_known_vals)
        for j in range(n):
            Xt[i][j] = (Xt[i][j] - mu)/sigma if Xt[i][j] != '' else 0

    return transpose(Xt)


def encode_features(raw_features):
    """
    returns an all-numeric feature matrix that is mean-normalized and scaled,
    converting non-numeric features to multiple numeric features as appropriate
    """

    X = [[] for i in range(len(raw_features))]
    for i, ex in enumerate(raw_features):
        # append Age,SibSp,Parch and Fare respectively
        for j in [4, 5, 6, 8]: X[i].append(float(ex[j]) if ex[j] else '')
        # encode Pclass
        X[i].extend([1, 0, 0] if ex[0] == '1' else [0, 1, 0] if ex[0] == '2' else [0, 0, 1] if ex[0] == '3' else ['', '', ''])
        # encode Sex
        X[i].extend([1, 0] if ex[3] == 'male' else [0, 1] if ex[3] == 'female' else ['', ''])
        # encode Embarked
        X[i].extend([1, 0, 0] if ex[10] == 'c' else [0, 1, 0] if ex[10] == 's' else [0, 0, 1] if ex[10] == 'q' else ['', '', ''])

    return standardize(X)


def parse_examples(filename):
    """
    extracts training examples from the titanic dataset. It returns a tuple consisting of two lists
    The first is the feature matrix - a list of feature vectors, one for each example. Non-numeric
    features have been encoded in a useful manner, and all features have been mean-normalized and
    scaled. The second list represents the 1-vector of associated ground truths.
    """
    y = []
    raw_features = []
    with open(filename) as f:
        data = list(f)
    for line in data:
        line = line.strip().lower().split(',')
        y.append(int(line[1]))
        raw_features.append(line[2:])
    # call encode_features to convert raw_features to a more useful feature matrix
    X = encode_features(raw_features)

    return (X, y)
