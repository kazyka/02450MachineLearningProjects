'''
Paths --
Toolbox path: ../02450 - Machine Learning/Toolbox/
Data path: ./insuranceCompany_Data/
    {'ticdata2000.txt', 'ticeval2000.txt', 'tictgts2000.txt'}
'''
import os
import sys
import numpy as np
import sklearn.preprocessing as skprep

# Add path to course toolbox
abs_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.dirname(abs_path)
par_par_path = os.path.dirname(par_path)
sys.path.append(par_par_path + '/02450 - Machine Learning/Toolbox/02450Toolbox_Python/Scripts')
sys.path.append(par_par_path + '/02450 - Machine Learning/Toolbox/02450Toolbox_Python/Tools')

# Check if data set exists
datafile = ['ticdata2000.txt', 'ticeval2000.txt', 'tictgts2000.txt']
for fname in datafile:
    if not os.path.isfile(par_path + '/insuranceCompany_Data/' + fname):
        print(fname+' is missing!', file=sys.stderr)
sys.stderr.flush()

## Load data from matlab file
X = np.loadtxt(par_path + '/insuranceCompany_Data/ticdata2000.txt')
X_eval = np.loadtxt(par_path + '/insuranceCompany_Data/ticeval2000.txt')
y_eval = np.loadtxt(par_path + '/insuranceCompany_Data/tictgts2000.txt')

y = X[:,-1]
X = X[:, 0:-1]

## Uncomment to balance class CARAVAN
# idx, keepIdx = np.where(y==0)[0], np.where(y==1)[0]
# under_R, over_R = 1.2, 2
# keepIdx = np.repeat(keepIdx, over_R) # Repeat positive class for over_R times
# chosenIdx = np.random.choice(idx, size=(int(under_R*len(keepIdx)),),
#     replace=False) # Reduce negative class to under_R times the positive class
# totalIdx = np.append(keepIdx,chosenIdx)
# totalIdx.sort()
# X = X[totalIdx,:]
# y = y[totalIdx]

## Uncoment to use one-out-of-k encoding
# For attr in [50,52]: [50, 51, 52] <=> [2, 3, 4]
# For attr > 52: Offset -48 to get the orignal attribute num (one-indexed)
one_out_of_k = [0,4]
for i in X:
    for j in one_out_of_k:
        i[j]-=1
for i in X_eval:
    for j in one_out_of_k:
        i[j]-=1
encoder = skprep.OneHotEncoder(categorical_features=one_out_of_k)
encoder.fit(X)
X = encoder.transform(X).toarray()
X_eval = encoder.transform(X_eval).toarray()

N = X.shape[0]
M = X.shape[1]
N_eval = X_eval.shape[0]
