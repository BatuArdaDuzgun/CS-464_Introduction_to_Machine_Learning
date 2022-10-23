
import csv
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm



from kPOD import k_pod

data_csv = pd.read_csv('movie_ID_with_ratings_with_mean.csv', header=None)
data = np.array(data_csv)
user_count = int(np.shape(data)[1] - 1) # 10638 # 3033
movie_count = int(np.shape(data)[0]) #3433 # 1058

number_of_clusters = 16

#kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=50, max_iter=500, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(data[:,1:])
#kmeans = SpectralClustering(n_clusters=number_of_clusters, assign_labels='kmeans', random_state=0).fit(data[:,1:])
kmeans = AgglomerativeClustering(n_clusters=number_of_clusters).fit(data[:,1:])






#code that draws the elbow plot:

distortions = []
K = range(2, 25)

for cluster_number in K:

    kmeanModel = KMeans(n_clusters=cluster_number, init='k-means++', n_init=200, max_iter=10000, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')

    kmeanModel.fit(data[:,1:])
    distortions.append(kmeanModel.inertia_)


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for K-means')
plt.show()
