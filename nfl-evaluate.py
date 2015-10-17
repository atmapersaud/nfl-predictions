import sys
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def main():
    testfile = sys.argv[1]
    modelfile = sys.argv[2]
    polyorder = int(sys.argv[3])
    testweeks = sys.argv[4]

    test_data = np.genfromtxt(testfile, delimiter=',', skip_header=1)

    X = test_data[:,:-1]
    y = test_data[:,-1]

    poly = PolynomialFeatures(degree=polyorder)
    Xpoly = poly.fit_transform(X)

    with open(modelfile, 'rb') as model, open(testweeks) as weeks:
        lr = pickle.load(model)
        games_per_week = (int(line) for line in weeks)
        ranges = []
        pos = 0
        for week in games_per_week:
            newpos = pos + week
            ranges.append( (pos, newpos) )
            pos = newpos
        print('W\tL\tPoints')
        weekly_results = (evaluate_week(week, Xpoly, y, lr) for week in ranges)
        for result in weekly_results:
            print('\t'.join(str(piece) for piece in result))

def evaluate_week(arr_range, data, labels, model):
    week_data = data[arr_range[0]:arr_range[1],:]
    week_labels = labels[arr_range[0]:arr_range[1]]

    week_preds = model.predict(week_data)
    week_probs = model.predict_proba(week_data)[:,1]
    probdevs = abs(week_probs-0.5)
    w, l, pts = 0, 0, 0

    order = probdevs.argsort()
    ranks = order.argsort()

    for i, pred in enumerate(week_preds):
        if pred == week_labels[i]:
            w += 1
            pts += ranks[i] + 1
        else:
            l += 1
    return (w, l, pts)

if __name__ == '__main__':
    main()
