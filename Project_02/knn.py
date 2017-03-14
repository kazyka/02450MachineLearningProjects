'''
KNeighborsClassifier
- Kfold cross validation for selecting k (# of neighbors)
'''
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, hist
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from resolve_path import *

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

# Maximum number of neighbors
L=50

# K-fold crossvalidation
K = 10
# CV = cross_validation.LeaveOneOut(N)

CV = cross_validation.KFold(X_train2.shape[0],K,shuffle=True)

errors = np.zeros((K,L))
k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X_train2[train_index,:], y_train2[train_index]
    X_test, y_test = X_train2[test_index,:], y_train2[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l, p=1);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[k,l-1] = np.sum(y_est!=y_test)
    k+=1

# Plot the classification error rate
figure()
plot(100*sum(errors,0)/len(y_train2))
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')

# Select the best model and plot
errors2 = np.zeros((L))
for l in range(1,L+1):
    knclassifier = KNeighborsClassifier(n_neighbors=l, p=1);
    knclassifier.fit(X_train2, y_train2);
    y_est = knclassifier.predict(X_test2);
    errors2[l-1] = np.sum(y_est!=y_test2)
figure()
plot(100*errors2/len(y_test2))
xlabel('\"K\" Nearest Neighbors')
ylabel('Classification error rate (%)')

show()
