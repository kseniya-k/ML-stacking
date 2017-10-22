import numpy as np

from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

import sklearn.svm


def stack_pred(estimator, X, y, Xt, k=3, method='predict'):
    method_to_call = getattr(estimator, method)
    kf = KFold(n_splits = k, shuffle = True, random_state = 0)
    
    pred_train = []
    pred_test = []
    for train_index, test_index in kf.split(X_all):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
        # Build j_th model, j = 1, ..., k
        estimator.fit(X_train, y_train)
        
        # Make prediction for train data
        pred = method_to_call(X)
        pred_train.append(pred)
    
        # Make prediction for test data
        pred = method_to_call(Xt)
        pred_test.append(pred)
        
    # Compute means through all k models' predictions
    result_train = np.mean(np.array(pred_train), axis = 0)
    result_test = np.mean(np.array(pred_test), axis = 0)
    
    return result_train.astype(int), result_test.astype(int)

# Data - 8*8 pictures of digits (0, ..., 9)
X_all, y_all = load_digits(return_X_y = True)
l = len(X_all)

classifier = sklearn.svm.SVC(gamma=0.001)


sX, sXt = stack_pred(classifier, X_all[:l/2], y_all[:l/2], X_all[l/2:], 6)

print accuracy_score(y_all[l/2:], sXt)
print accuracy_score(y_all[:l/2], sX)