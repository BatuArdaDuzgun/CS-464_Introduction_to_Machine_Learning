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


number_of_clusters = 9 # must be manualy updated for the given clustering


data_csv = pd.read_csv('k-means_clustering.csv', header=None)
predictions = np.array(data_csv)


# this code draws the genre dsitirbutions between
movie_csv = pd.read_csv('movie.csv')
movies = np.array(movie_csv)

posible_genres = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir', '(no genres listed)'] #list off all the genres.

genres_distribution = np.zeros((number_of_clusters, (len(posible_genres)+1))) #each row is for a cluster. colums are for different genre counts. the last one is the total count for calculating percentage at the end.


rating_csv = pd.read_csv('new_rating.csv', header=None)



for cluster_ID in range(number_of_clusters):


    ratings = np.array(rating_csv)
    ratings = ratings[:, 1:]  # drop first column
    ratings = ratings[ratings[:, 0].argsort()]  # sort to first movie Id

    movieIds = predictions[np.where(predictions[:, 1] == cluster_ID)][:, 0]  # movieIdList of 0. cluster


    arr = np.array([])
    for i, ind in enumerate(movieIds):
        arr = np.append(arr, ratings[np.where(ratings[:, 0] == movieIds[i])].sum(axis=0)[1])

    if np.shape(arr)[0] >= 12:

        prominent_movie_IDs = movieIds[arr.argsort()[::-1][:12]]

    else:

        prominent_movie_IDs = movieIds[arr.argsort()]

    prominent_movie_names = []

    for movie_ID in prominent_movie_IDs:

        index = np.where(movies[:,0] == movie_ID)

        name = movies[index, 1]

        prominent_movie_names.append(name[0])

    #printing the results
    print("\n\nClulster " + str(cluster_ID)+ " Prominent Movies:\n")

    for i in range(len(prominent_movie_names)):

        print(str(i + 1) + ".) " + prominent_movie_names[i])









problem_count = 0





for i in range(np.shape(predictions)[0]):

    movie_index = int(predictions[i, 0] - 1)
    movie_cluster = int(predictions[i, 1])

    if movie_index >= 27278: # movie IDs without entries in the movies array.

        problem_count += 1
        continue

    movie_genres = movies[movie_index, 2]

    movie_genres_list = movie_genres.split("|")

    for genre in movie_genres_list:

        genre_index = posible_genres.index(genre)

        genres_distribution[movie_cluster, genre_index] += 1 # counting each genre tag one by one

    genres_distribution[movie_cluster, 20] += 1 # increasing the total movie count in the cluster by one


for i in range(np.shape(genres_distribution)[0]):

    genres_distribution[i, :-1] /= genres_distribution[i, -1]

for i in range(np.shape(genres_distribution)[1] - 1):

    genres_distribution[:, i] /= sum(genres_distribution[:, i])

genres_distribution[:,-2] = 0 # there are only 1 or 2 of them making the plot look ugly



plt.imshow(genres_distribution[:, :-1], cmap='hot')
#plt.yticks(genres_distribution[:, -1])


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y = range(number_of_clusters)

plt.xticks(x, posible_genres, rotation='vertical')
plt.yticks(y, list(genres_distribution[:, -1]))

plt.xlabel("Genre Names")
plt.ylabel("Number of Movies in Cluster")

plt.title("Percent of Movies With the Genre in the Cluster Over Percent of Movies With the Genre in All Cluster")

plt.show()




for i in range(np.shape(genres_distribution)[1]):

    genres_distribution[:, i] /= sum(genres_distribution[:, i])

for i in range(np.shape(genres_distribution)[0]):

    genres_distribution[i,:-1] /= genres_distribution[i,-1]





