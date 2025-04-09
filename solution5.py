## Exercise 5 (10 minutes): Evaluating Clusters with Silhouette Scores
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load or assume you have a preprocessed dataset (df_scaled)
#    For demonstration, we'll again load & scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Get the cluster labels from each method
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
agg_labels = agg.labels_

# 4. Compute silhouette scores (only if more than one cluster exists)
#    DBSCAN might produce a single cluster or no clusters if parameters are not well-tuned,
#    so we check to avoid an error in silhouette_score.
def calculate_silhouette_score(labels, X):
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1

kmeans_silhouette = calculate_silhouette_score(kmeans_labels, X_scaled)
dbscan_silhouette = calculate_silhouette_score(dbscan_labels, X_scaled)
agg_silhouette = calculate_silhouette_score(agg_labels, X_scaled)

# 5. Print the scores
print("KMeans silhouette score", kmeans_silhouette)
print("DBSCAN silhouette score", dbscan_silhouette)
print("Agglomerative Clustering silhouette score", agg_silhouette)
