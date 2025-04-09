## Exercise 4 (10 minutes): Agglomerative Clustering & Dendrogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we simulate df_scaled by loading and scaling the Iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# ------------------------------------------------------------------------------------

# 2. Perform Agglomerative Clustering
aggClust = AgglomerativeClustering(n_clusters=3, linkage='ward')

# 3. Add the cluster labels to the DataFrame
aggClust_labels = aggClust.fit_predict(df_scaled)
df_scaled['Agglomerative_Cluster'] = aggClust_labels

# 4. Print a quick summary of how many points were assigned to each cluster
cluster_summary = df_scaled['Agglomerative_Cluster'].value_counts()
print("Distribution - Agglomerative Clustering:")
for cluster, count in cluster_summary.items():
    print(f"Cluster {cluster}: {count} points")

# 5. Create a linkage matrix for plotting a dendrogram
#    Note: We exclude the 'cluster' column when computing the linkage
linkage_matrix = linkage(df_scaled, method='ward')

# 6. Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Agglomerative Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
