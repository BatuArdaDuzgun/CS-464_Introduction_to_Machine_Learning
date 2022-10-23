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

number_of_clusters = 9

kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=300, max_iter=10000, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(data[:,1:])
#kmeans = SpectralClustering(n_clusters=number_of_clusters, assign_labels='kmeans', random_state=0).fit(data[:,1:])
#kmeans = AgglomerativeClustering(n_clusters=number_of_clusters).fit(data[:,1:])

classes = kmeans.labels_



predictions = np.zeros((movie_count, 2))

predictions[:, 0] = data[:, 0]



predictions[:, 1] = classes


def write_csv(new_data):
    with open('k-means_clustering.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(new_data)


write_csv(predictions)