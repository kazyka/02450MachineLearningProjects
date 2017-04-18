'''
Decision Tree, Logistic Regression, Naive bayes, KNN
- Plot model performace for each classifier
- Select optimal model for each classifier (marked by red vertical line)
- Plot confusion matrix for each optimal model
- TODO: do one-out-of-k encoding
'''
import numpy as np
from sklearn import cross_validation
from resolve_path import *

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)
errors = {}
errors['random'] = np.zeros((K))

k=0
print("\t\t\t\t\t\tRand")
for train_index, test_index in CV:

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    m_errors = []

    # Fit KNN classifier and classify the test points (consider 1 to L neighbors)
    y_est = np.random.rand(y_test.shape[0])
    residuals = (y_est-y_test).astype(float)
    errors['random'][k] = np.dot(residuals,residuals)/y_test.shape[0]
    k+=1


# Second-level validation (Predict on ticeval2000.txt)
g_errors = {}

y_est = np.random.rand(y_eval.shape[0])
residuals = (y_est-y_eval).astype(float)
g_errors['random'] = np.dot(residuals,residuals)/N_eval

print('\n\t\t\tCross-validation error:\t{0:.3f}'.format(
        np.mean(errors['random'])))

print('\n\t\t\tGeneralization error:\t{0:.3f}'.format(
        g_errors['random']))
