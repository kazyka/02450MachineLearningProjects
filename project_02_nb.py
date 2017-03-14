'''
MultinomialNB (Naive Bayes)
- KFold cross validation for train error calculation
'''
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

# Load data from matlab file
X = np.loadtxt('./insuranceCompany_Data/ticdata2000.txt')
y = X[:,-1]
X = X[:, 0:-1]

N = X.shape[0]
M = X.shape[1]

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

# Naive Bayes classifier parameters
alpha = 1.0         # additive parameter (e.g. Laplace correction)
est_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(X_train2.shape[0],K,shuffle=True)
errors = np.zeros(K)
k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X_train2[train_index,:], y_train2[train_index]
    X_test, y_test = X_train2[test_index,:], y_train2[test_index]

    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=True)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)

    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
    k+=1

# Plot the classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))
