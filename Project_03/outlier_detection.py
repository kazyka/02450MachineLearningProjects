from resolve_path import *
import matplotlib.pyplot as plt
import numpy as np
from toolbox_02450 import clusterplot, gausKernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

def dist2(X, v, verbose=False):
    mat = np.zeros(len(v))
    for m_idx,elem in enumerate(v):
        mat[m_idx] = np.sqrt(np.square(X[v[0]]-X[elem]).sum())
    if verbose:
        for idx,d in enumerate(mat):
            print('[{0}/{1}] dist(X[{2}],X[{3}]) = {4:.20f}'.format(idx,len(mat),v[0],v[idx],mat[idx]))
    return mat
def printOutliers(d, d_idx, n=10):
    print('Top {0} outliers(score):'.format(n))
    for i in range(n):
        print('\t[{0}/{1}] X[{2}]({3:.6f})'.format(i+1,n,d_idx[i],d[i]))
def printOutliersLatex(d, d_idx, n=20):
    for i in range(n):
        print('{0} & {1} & {2:.6f} & {3} & {4:.6f} & {5} & {6:.6f} \\\\ \\hline'
            .format(i+1, d_idx['kde'][i], np.log(d['kde'][i]), d_idx['knn'][i], d['knn'][i], d_idx['ard'][i], d['ard'][i]))
def drawOutlierBar(n_outlier):
    plt.figure()
    plt.subplot(3,1,1)
    plt.bar(range(n_outlier),d['kde'][:n_outlier])
    plt.title('Gausian Kernel density: Outlier score')
    plt.subplot(3,1,2)
    plt.bar(range(n_outlier),d['knn'][:n_outlier])
    plt.title('KNN density: Outlier score')
    plt.subplot(3,1,3)
    plt.bar(range(n_outlier),d['ard'][:n_outlier])
    plt.title('KNN average relative density: Outlier score')
    plt.show()

d = {}
d_idx = {}

### Attribute normalization
X = X / np.max(X, axis=0)

print('Calculating Gaussian Kernel density...')
### Gausian Kernel
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()

val = logP.max()
ind = logP.argmax()

width = widths[ind]
# width = 0.32228417991810204
print('\tOptimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
d['kde'], log_density = gausKernelDensity(X,width)

# Sort the densities
d_idx['kde'] = (d['kde'].argsort(axis=0)).ravel()
d['kde'] = d['kde'][d_idx['kde']].reshape(d['kde'].shape[0])

printOutliers(np.log(d['kde']), d_idx['kde'])
print('Calculating K-Nearest Neighbors density...')
### K-neighbors density estimator
K = 150
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

for v_idx, v in enumerate(i):
    D[v_idx] = dist2(X,v)

d['knn'] = 1./(D.sum(axis=1)/K)

# Sort the scores
d_idx['knn'] = d['knn'].argsort()
d['knn'] = d['knn'][d_idx['knn']].reshape(d['knn'].shape[0])

printOutliers(d['knn'], d_idx['knn'])
print('Calculating K-Nearest Neighbors average relative density (ARD)...')
### K-nearest neigbor average relative density
d['ard'] = d['knn']/(d['knn'][i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
d_idx['ard'] = d['ard'].argsort()
d['ard'] = d['ard'][d_idx['ard']].reshape(d['ard'].shape[0])

printOutliers(d['ard'], d_idx['ard'])
printOutliersLatex(d,d_idx)

drawOutlierBar(80)
