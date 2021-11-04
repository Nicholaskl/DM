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
from sklearn.metrics.pairwise import pairwise_distances_argmin

df = pd.read_csv('bmw-browsers.csv')
X = df.values

num_k = 5
# run kmeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
kmeans = KMeans(n_clusters=num_k,random_state=0).fit(X)

# extract centroids
clusters = kmeans.cluster_centers_

# instance counts in each cluster
counts = np.bincount(kmeans.labels_)

colors = ["#4EACC5", "#FF9C34", "#4E9A06", "#BB34FF", "#FF346E"]

k_means_labels = pairwise_distances_argmin(X, clusters)

fig = plt.figure(figsize=(8, 3))

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(num_k), colors):
    my_members = k_means_labels == k
    cluster_center = clusters[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor="#bbbbbb", marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
# %%
