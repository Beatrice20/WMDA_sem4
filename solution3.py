## Exercise 3 (10 minutes): DBSCAN Clustering
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we'll again simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------
print(iris.feature_names)

# 2. Instantiate DBSCAN with chosen parameters
#    eps defines the neighborhood radius, min_samples is the minimum number of points
#    for a region to be considered dense.
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
outliers = np.sum(labels == -1)
print(f"Number of outliers: {outliers}")

# 6. (Optional) Add the labels to the DataFrame
df_scaled['Cluster_DBSCAN'] = labels

# 7. Print the cluster label counts
unique_labels, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique_labels, counts))
print("\nCluster counts:")
print(cluster_counts)

# 8. Optional quick visualization (for 2D only)
#    Choose two features to plot, coloring by DBSCAN labels
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('DBSCAN - Scatter Plot')
plt.colorbar(label='Cluster Label')
plt.show()