# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os

# 1. Load and clean data
df = pd.read_csv('../processed_data.csv')
df = df.dropna()

# 2. Normalize data (scale to 0-1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.select_dtypes('number'))

# 3. K-Means Clustering (Simple Groups)
kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_group'] = kmeans.fit_predict(normalized_data)

# 4. Hierarchical Clustering (Tree Groups)
Z = linkage(normalized_data, 'ward')  # Creates family tree of data

# Save dendrogram plot
plt.figure(figsize=(10,5))
dendrogram(Z)
plt.savefig('../results/agnes_dendrogram.png')
plt.close()

# Get final groups from tree
df['hierarchical_group'] = fcluster(Z, t=3, criterion='maxclust')

# 5. Save results
df.to_csv('../results/clusters.csv', index=False)
print("Done! Check the results folder!")