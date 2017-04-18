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
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from scipy.stats import t
from resolve_path import *

### Initialize parameters used in the classifiers
# Linear regression
A = M-1

# ANN
L = 10

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)
errors = {}
errors['lin'] = np.zeros((K,A))
errors['ann'] = np.zeros((K,L))

k=0
print("\t\t\t\t\t\tLin\tANN")
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}\t'.format(k+1,K), end="")

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    m_errors = []

    # Fit KNN classifier and classify the test points (consider 1 to L neighbors)
    for i in range(0,A):
        lin = LinearRegression()
        lin.fit(X_train[:,0:i+1], y_train)
        y_est = lin.predict(X_test[:,0:i+1])
        residuals = (y_est-y_test).astype(float)
        errors['lin'][k,i] = np.dot(residuals,residuals)/y_test.shape[0]
    m_errors.append(np.min(errors['lin'][k]))

    for i in range(0,L):
        ann = MLPRegressor(solver='adam',alpha=1e-4,hidden_layer_sizes=(i+1,),random_state=0,max_iter=200,activation='relu')
        ann.fit(X_train, y_train)
        y_est = ann.predict(X_test)
        residuals = (y_est-y_test).astype(float)
        errors['ann'][k,i] = np.dot(residuals,residuals)/y_test.shape[0]
    m_errors.append(np.min(errors['ann'][k]))

    print('Minimum error:\t{0:.3f}\t{1:.3f}'.format(*m_errors))
    k+=1

# Model selection
a = np.argmin(np.mean(errors['lin'],0))+1
lin = LinearRegression()

l = np.argmin(np.mean(errors['ann'],0))+1
ann = MLPRegressor(solver='adam',alpha=1e-4,hidden_layer_sizes=(l,),random_state=0,max_iter=200,activation='relu')

# Second-level validation (Predict on ticeval2000.txt)
g_errors = {}

lin.fit(X[:,0:a],y)
y_est = lin.predict(X_eval[:,0:a])
residuals = (y_est-y_eval).astype(float)
g_errors['lin'] = np.dot(residuals,residuals)/N_eval

ann.fit(X,y)
y_est = ann.predict(X_eval)
residuals = (y_est-y_eval).astype(float)
g_errors['ann'] = np.dot(residuals,residuals)/N_eval

plt.figure()
plt.plot(np.arange(start=1,stop=A+1,step=1),100*sum(errors['lin'],0)/N)
plt.axvline(x=a, linewidth=2, color='r',linestyle='--')
plt.ylim(ymin=0.2,ymax=0.5)
plt.title('Linear regression')
plt.xlabel('Number of attributes')
plt.ylabel('Mean squared error')

plt.figure()
plt.plot(np.arange(start=1,stop=L+1,step=1),100*sum(errors['ann'],0)/N)
plt.axvline(x=l, linewidth=2, color='r',linestyle='--')
plt.ylim(ymin=0.2,ymax=0.5)
plt.title('Linear regression with ANN')
plt.xlabel('Number of attributes')
plt.ylabel('Mean squared error')


print('\n\t\t\tGeneralization error:\t{0:.3f}\t{1:.3f}'.format(
        g_errors['lin'],g_errors['ann']))

dif = errors['lin'][:,a]-errors['ann'][:,l]
mean = np.mean(dif,0)
sd = np.std(dif,0)
df = K-1
p_interval = np.zeros((2))
p_interval[0] = mean-t.ppf(0.975,df)*(sd*np.sqrt(K))
p_interval[1] = mean+t.ppf(0.975,df)*(sd*np.sqrt(K))
print('\nConfidence interval for t test: ',p_interval)

plt.show()

