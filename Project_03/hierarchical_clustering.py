# exercise 10.2.1
from resolve_path import *

from matplotlib.pyplot import figure, show
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Load Matlab data file and extract variables of interest
N, M = X.shape

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'ward'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 10
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1))

# Display dendrogram
max_display_levels=10
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
