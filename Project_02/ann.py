import numpy as np
import sklearn.neural_network as nn

from sklearn import cross_validation
from resolve_path import *

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

# K-fold crossvalidation
K = 10
l1_max_nodes = 5

CV = cross_validation.KFold(X_train2.shape[0],K,shuffle=True)
#errors = np.zeros((K,l1_max_nodes-1,l2_max_nodes))
errors = np.zeros((K,l1_max_nodes))
k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X_train2[train_index,:], y_train2[train_index]
    X_test, y_test = X_train2[test_index,:], y_train2[test_index]

    for i in range(0,l1_max_nodes):
    	dnn_classifier = nn.MLPClassifier(solver='adam',alpha=1e-4,hidden_layer_sizes=(i+1,),random_state=0,max_iter=200,activation='tanh')
    	dnn_classifier.fit(X_train, y_train)
    	y_est = dnn_classifier.predict(X_test)
    	errors[k,i] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]

    k+=1

# Plot the classification error rate
print('Error rate: ',100*np.mean(errors,0))
