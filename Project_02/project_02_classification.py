'''
Decision Tree, Logistic Regression, Naive bayes, KNN
- Plot model performace for each classifier
- Select optimal model for each classifier (marked by red vertical line)
- Plot confusion matrix for each optimal model
- TODO: do one-out-of-k encoding
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from resolve_path import *

### Initialize parameters used in the classifiers
## DecisionTreeClassifier
M = 20
Tc = np.arange(2, M+2, 1)
criterion = 'gini'

## LogisticRegression
C = 12  # Inverse of regularization strength 10^-(C/2) : 10^(C/2)

## MultinomialNB
alpha = 1.0
Prior = [sum(y==0)/N, 1-sum(y==0)/N]

## KNeighborsClassifier
L = 20  # Maximum number of neighbors
p = 2   # Distance measure (1: Manhattan, 2: Euclidean)

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)
errors = {}
errors['dtc'] = np.zeros((K,M))
errors['log'] = np.zeros((K,C+1))
errors['nb'] = np.zeros((K,2))
errors['knn'] = np.zeros((K,L))

k=0
print("\t\t\t\t\t\tDtc\tLog\tNB\tNBw/P\tKNN\tANN")
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}\t'.format(k+1,K), end="")

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    m_errors = []

    for i, t in enumerate(Tc):
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=t)
        dtc = dtc.fit(X_train, y_train)
        y_est = dtc.predict(X_test)
        errors['dtc'][k,i] = np.sum(y_est!=y_test)
    m_errors.append(np.mean(errors['dtc'][k]))

    for c in range(C+1):
        log = LogisticRegression(C=pow(10,c-C/2))
        log = log.fit(X_train, y_train)
        y_est = log.predict(X_test)
        errors['log'][k,c] = np.sum(y_est!=y_test)
    m_errors.append(np.mean(errors['log'][k]))

    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=True)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)
    errors['nb'][k,0] = np.sum(y_est!=y_test)
    m_errors.append(errors['nb'][k,0])

    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=True, class_prior=Prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)
    errors['nb'][k,1] = np.sum(y_est!=y_test)
    m_errors.append(errors['nb'][k,1])

    # Fit KNN classifier and classify the test points (consider 1 to L neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l, p=p);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors['knn'][k,l-1] = np.sum(y_est!=y_test)
    m_errors.append(np.mean(errors['knn'][k]))
    print('Average errors:\t{0:.3f}\t{1:.3f}\t{2:.1f}\t{3:.1f}\t{4:.3f}'.format(*m_errors))
    k+=1

# Model selection
t = Tc[np.argmin(np.sum(errors['dtc'], 0))]
dtc = DecisionTreeClassifier(criterion=criterion, max_depth=t)

c = np.argmin(np.sum(errors['log'], 0))
log = LogisticRegression(C=pow(10,c-C/2))

prior = Prior if np.sum(errors['nb'], 0)[0] > np.sum(errors['nb'], 0)[1] else None
nb_classifier = MultinomialNB(alpha=alpha, fit_prior=True, class_prior=prior)

l = np.argmin(np.sum(errors['knn'], 0))
knclassifier = KNeighborsClassifier(n_neighbors=l, p=p);

# Second-level validation (Predict on ticeval2000.txt)
g_errors = {}
cm = {}

dtc.fit(X, y);
y_est = dtc.predict(X_eval);
g_errors['dtc'] = np.sum(y_est!=y_eval)/N_eval
cm['dtc'] = confusion_matrix(y_eval, y_est);

log.fit(X, y);
y_est = log.predict(X_eval);
g_errors['log'] = np.sum(y_est!=y_eval)/N_eval
cm['log'] = confusion_matrix(y_eval, y_est);

nb_classifier.fit(X, y);
y_est = nb_classifier.predict(X_eval);
g_errors['nb'] = np.sum(y_est!=y_eval)/N_eval
cm['nb'] = confusion_matrix(y_eval, y_est);

knclassifier.fit(X, y);
y_est = knclassifier.predict(X_eval);
g_errors['knn'] = np.sum(y_est!=y_eval)/N_eval
cm['knn'] = confusion_matrix(y_eval, y_est);

plt.figure()
plt.plot(100*sum(errors['dtc'],0)/N)
plt.axvline(x=t, linewidth=2, color='r')
plt.title('Decision Tree Classifier')
plt.xlabel('Tree depth')
plt.ylabel('Classification error rate (%)')

plt.figure()
plt.plot(100*sum(errors['log'],0)/N)
plt.axvline(x=c, linewidth=2, color='r')
plt.title('Logistic Regression Classifier')
plt.xlabel('c (C = 1/lambda = 1/10^c)')
plt.ylabel('Classification error rate (%)')

plt.figure()
plt.plot(100*sum(errors['knn'],0)/N)
plt.axvline(x=l, linewidth=2, color='r')
plt.title('KNN Classifier')
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')

print('\n\t\t\tGeneralization error:\t{0:.3f}\t{1:.3f}\t{2:.3f}\t\t{3:.3f}'.format(
        g_errors['dtc'], g_errors['log'], g_errors['nb'], g_errors['knn']))
for k, v in cm.items():
    accuracy = 100*v.diagonal().sum()/v.sum(); error_rate = 100-accuracy;
    print('Classifier {0}:'.format(k))
    print('\tAccuracy / ErrorRate\t{0:.4f}\t\t{1:.4f}'.format(accuracy, error_rate))
    print('\tConfusion Matrix\tTP={0}\t\tTN={1}\t\tFP={2}\t\tFN={3}'.format(v[1,1],v[0,0],v[0,1],v[1,0]))
    plt.figure()
    plt.imshow(v, cmap='binary', interpolation='None'); plt.colorbar();
    plt.xticks([0, 1]); plt.yticks([0, 1]);
    plt.xlabel('Predicted class'); plt.ylabel('Actual class');
    plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

plt.show()
