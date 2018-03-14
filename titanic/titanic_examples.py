def encode_features(raw_features):
    """
    returns an all-numeric feature matrix that is mean-normalized and scaled,
    converting non-numeric features to multiple numeric features as appropriate
    """
    X = []
    for i,ex in enumerate(raw_features):
        #encode Pclass
        X[i].extend(1,0,0) if ex[0] == '1' else X[i].extend(0,1,0) if ex[0] == '2' else X[i].extend(0,0,1)
        #encode male/female - use 50/50 avg for now if gender unknown
        X[i].extend(1,0) if ex[3] == 'male' else X[i].extend(0,1) if ex[3] == 'female' else X[i].extend(.5,.5)
    return raw_features

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
        line = line.split(',')
        y.append(int(line[1]))
        raw_features.append(line[2:])
    #call encode_features to convert raw_features to a more useful feature matrix
    X = encode_features(raw_features)
    return (X,y)