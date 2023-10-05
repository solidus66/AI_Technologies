import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file_path = 'computer.dat'
data = pd.read_csv(file_path, sep='\t')

selected_variables = ['V1A', 'V1B', 'V1C', 'V2A', 'V2B', 'V2C', 'V2D', 'V2E', 'V2F', 'V3A', 'V3B', 'V3C', 'V4A', 'V4B',
                      'V4C', 'V4D', 'V4E', 'V5A', 'V5B']
X = data[selected_variables]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, show_leaf_counts=True, orientation='top', no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')  # Дендрограмма иерархической кластеризации
plt.show()

distances = linkage_matrix[:, 2]
deltas = distances[:-1] - distances[1:]
plt.plot(range(1, len(deltas) + 1), deltas, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distance Change')
plt.title('Elbow Method for Hierarchical Clustering')  # Метод "каменистая осыпь" для иерархической кластеризации
plt.show()

distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for K-Means Clustering')  # Метод "каменистая осыпь" для метода k средних
plt.show()

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X_scaled)

data['Cluster_kmeans'] = kmeans.labels_

mds = MDS(n_components=2, random_state=42, n_init=10, normalized_stress=False)
X_mds = mds.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(X_mds[data['Cluster_kmeans'] == i, 0], X_mds[data['Cluster_kmeans'] == i, 1], label=f'Cluster {i + 1}',
                s=50)
plt.title('Visualization of K-Means Clustering Results with MDS')  # Визуализация результатов метода k средних с MDS
plt.legend()
plt.show()
