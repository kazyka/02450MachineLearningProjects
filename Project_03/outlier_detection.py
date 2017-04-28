from resolve_path import *
print(sys.path)

from matplotlib.pyplot import figure, show
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture


K = 1 # Leave-one-out

cov_type = 'diag'
# type of covariance, you can try out 'diag' as well
reps = 5
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type == 'diag':
    new_covs = np.zeros([K,M,M])

count = 0
for elem in covs:
    temp_m = np.zeros([M,M])
    for i in range(len(elem)):
        temp_m[i][i] = elem[i]

    new_covs[count] = temp_m
    count += 1

covs = new_covs
# Plot results:
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()
