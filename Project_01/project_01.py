from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title,
yticks, show,legend,imshow, cm, hold, grid)
from scipy.linalg import svd
import numpy as np
import resolve_path

# Load the data
X = np.loadtxt('./insuranceCompany_Data/ticdata2000.txt')

N = X.shape[0]
M = X.shape[1]

# Subtract mean value from data
Xc = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Xc
U,S,VT = svd(Xc,full_matrices=False)
V = VT.T
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
for i in range(len(V)):
    v = V[:,i]
    if v.argmax() >= 43:
        print(i, v.argmax(), v.max())
    if v.argmin() >= 43:
        print(i, v.argmin(), v.min())
# Plot Cumulative variance explained
cumu = []
sumS = 0
for i in rho:
    sumS+=i
    cumu.append(sumS)
figure()
grid()
plot(range(1,len(cumu)+1),cumu,'o-')
title('Cumulative variance explained by the first N principal components');
xlabel('First N principal components');
ylabel('Variance explained');

# Plot variance explained
figure()
grid()
plot(range(1,len(V[:,0])+1),V[:,0],'-', color='#3383ba')
plot(range(1,len(V[:,1])+1),V[:,1],'-', color='red')
plot(range(1,len(V[:,2])+1),V[:,2],'-', color='yellowgreen')
plot(range(1,len(V[:,3])+1),V[:,3],'-', color='magenta')
title('Direction of the first 4 principal components')
xlabel('Attributes')
legend(['PC1','PC2','PC3', 'PC4'])

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');

# Selected number of principal components
k = 13

# Project the data onto the considered principal components
Z = Xc @ V[:,:k]

# Plot PCA of the data
nb_classes = np.array([0,1])
classNames = ['c0','c1']
style = ['bo', 'ro']
f = figure()
f.hold()
title('Insurance Company Data: PCA')
for c in nb_classes:
    # select indices belonging to class c:
    class_mask = X[:,85] == c
    plot(Z[class_mask,0], Z[class_mask,1], style[c])
legend(classNames)
xlabel('PC1')
ylabel('PC2')


# Find variance-covariance matrix and correlation
covVarMatrix = np.cov(Xc.T)
corMatrix = np.corrcoef(Xc.T) - np.diag(np.ones(M))
print(corMatrix.argmax(),corMatrix.max())

# Show the plots
show()
