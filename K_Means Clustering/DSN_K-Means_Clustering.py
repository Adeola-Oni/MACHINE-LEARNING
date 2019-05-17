#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#get dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

#use elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10, max_iter = 300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying KMeans to the dataset
kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans= kmeans.fit_predict(X)

#visualizing the clusters 
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red', label='cluster1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue', label='cluster2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green', label='cluster3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c='purple', label='cluster4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=100, c='orange', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()