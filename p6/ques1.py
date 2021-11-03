#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bmw-browsers.csv')

k = 5

# run kmeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
kmeans = KMeans(n_clusters=k,random_state=0).fit(df)

# extract centroids
clusters = kmeans.cluster_centers_

# instance counts in each cluster
counts = np.bincount(kmeans.labels_)
df.plot.scatter('x', 'y')
# plt.scatter(df[:, 0], df[:, 1], c=kmeans)


# %%
