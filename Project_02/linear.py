'''
LogisticRegression
- KFold cross validation for train error calculation
'''
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, axhline
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation, tree
from resolve_path import *

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(X_train2.shape[0],K,shuffle=True)

errors = np.empty((K, 1))

k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X_train2[train_index,:], y_train2[train_index]
    X_test, y_test = X_train2[test_index,:], y_train2[test_index]

    # Fit and evaluate Logistic Regression classifier
    mod = lm.LinearRegression()
    mod = mod.fit(X_train, y_train)
    y_est = mod.predict(X_test)
    errors[k] = np.dot((y_est-y_test).astype(float),(y_est-y_test).astype(float))/len(y_test)
    k+=1

print('Error rate: ',100*np.mean(errors,0))
figure();
plot(100*errors)
axhline(y=100*np.mean(errors), xmin=0, xmax=1, hold=None, color="red")
title('Linear Regression')
xlabel('n Fold')
ylabel('Mean square error rate at fold n(%)')
show()
