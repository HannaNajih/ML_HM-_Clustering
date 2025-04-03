import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import io

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'processed_data.csv')
df = pd.read_csv(data_path)

# 2. Handle missing values
print("\n=== Missing Values ===")
print("Before cleaning:", df.isnull().sum().sum(), "missing values")
df = df.dropna()
print("After cleaning:", df.isnull().sum().sum(), "missing values")

# 3. Normalize and cluster
print("\n=== Clustering ===")
scaler = MinMaxScaler()
numeric_data = df.select_dtypes(include=['int64', 'float64'])
normalized_data = scaler.fit_transform(numeric_data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(normalized_data)

# 4. Save results
results_path = os.path.join(current_dir, '..', 'results', 'clusters.csv')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
df.to_csv(results_path, index=False)

print("\n✅ Success! Results saved to:", results_path)
print("Cluster distribution:")
print(df['cluster'].value_counts())