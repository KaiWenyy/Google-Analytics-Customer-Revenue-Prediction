from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def apply_kmeans(x_train, x_test):
	kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
	print("Labels", kmeans.labels_)

	y_test = kmeans.predict(x_test)
	print("Labels", y_test)
	print("centers =", kmeans.cluster_centers_)


def apply_hierarchical(x_train, x_test):
	clustering = AgglomerativeClustering().fit(x_train)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
	print(kmeans.labels_)
	print("Labels", clustering.labels_)

	# Visualization tutorial: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

