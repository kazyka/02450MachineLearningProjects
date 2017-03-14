import numpy as np
import sklearn.linear_model as lm
from pylab import *

import sys
path_to_lib = '/home/martin/Dropbox/DTU/6. Semester/02450 - Machine Learning/Toolbox/02450Toolbox_Python/Scripts'
sys.path.insert(0, path_to_lib)

# Load the data
data = np.loadtxt('../insuranceCompany_Data/ticdata2000.txt')

M,N = data.shape

# classification variables
y_class = data[:,-1]
X_class = data[:,:-2]

model_class = lm.LogisticRegression()
model_class = model_class.fit(X_class,y_class)
y_class_est = model_class.predict(X_class)
y_class_est_prob = model_class.predict_proba(X_class)[:, 0]
misclass_rate = sum(np.abs(np.mat(y_class_est).T - y_class)) / float(len(y_class_est))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))


f = figure(); f.hold(True)
class0_ids = nonzero(y_class==0)[0].tolist()
plot(class0_ids, y_class_est_prob[class0_ids], '.y')
class1_ids = nonzero(y_class==1)[0].tolist()
plot(class1_ids, y_class_est_prob[class1_ids], '.r')
xlabel('Data object (CARAVAN)'); ylabel('Predicted prob. of class CARAVAN');
legend(['0', '1'])
ylim(-0.2,1.2)

# regression variables
y_reg = data[:,0]
X_reg = data[:,1:]

model_reg = lm.LinearRegression()
model_reg = model_reg.fit(X_reg,y_reg)
y_reg_est = model_reg.predict(X_reg)
print('Regression MSE: ',np.dot(y_reg-y_reg_est,y_reg-y_reg_est)/M)

# Plot original data and the model output
f = figure()
f.hold(True)
plot(X_reg,y_reg,'.')
plot(X_reg,y_reg_est,'-')
xlabel('X'); ylabel('y'); ylim(-2,8)
legend(['Training data', 'Regression fit (model)'])

show()
