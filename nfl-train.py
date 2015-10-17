import sys
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def main():
    trainfile = sys.argv[1]
    modelfile = sys.argv[2]
    polyorder = int(sys.argv[3])
    
    # read in training data
    train_data = np.genfromtxt(trainfile, delimiter=',', skip_header=1)

    X = train_data[:,:-1]
    y = train_data[:,-1]

    poly = PolynomialFeatures(degree=polyorder)
    Xpoly = poly.fit_transform(X)
    
    lr = LogisticRegression()

    lr.fit(Xpoly, y)

    with open(modelfile, 'wb') as f:
        pickle.dump(lr, f)
    
if __name__ == '__main__':
    main()
