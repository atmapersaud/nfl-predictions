import sys
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def getWinnerProb(item):
    return item[1]

def main():
    weekfile = sys.argv[1]
    modelfile = sys.argv[2]
    polyorder = int(sys.argv[3])
    metadata = sys.argv[4]

    week_data = np.genfromtxt(weekfile, delimiter=',', skip_header=1)
    
    poly = PolynomialFeatures(degree=polyorder)
    Xpoly = poly.fit_transform(week_data)

    with open(modelfile, 'rb') as model, open(metadata) as md:
        lr = pickle.load(model)
        preds = lr.predict(Xpoly).astype(int)
        probs = lr.predict_proba(Xpoly)
        results = []

        for i, line in enumerate(md):
            home, away = line.strip().split(',')
            if preds[i] == 1:
                results.append((home+'*', "{0:.3f}".format(probs[i,1]), away, "{0:.3f}".format(probs[i,0])))
            else: 
                results.append((away, "{0:.3f}".format(probs[i,0]), home+'*', "{0:.3f}".format(probs[i,1])))

        results = sorted(results, key=getWinnerProb, reverse=True)
        for result in results:
            print('\t'.join(result))
        
if __name__ == '__main__':
    main()
