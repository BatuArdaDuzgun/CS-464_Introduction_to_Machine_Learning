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

kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=50, max_iter=1000, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(data[:,1:])
#kmeans = SpectralClustering(n_clusters=number_of_clusters, assign_labels='kmeans', random_state=0).fit(data[:,1:])
#kmeans = AgglomerativeClustering(n_clusters=number_of_clusters).fit(data[:,1:])

classes = kmeans.labels_
"""
# Kpops code:

# set the number of clusters to 3
K = number_of_clusters

# use data with missing values to perform clustering
clustered_data = k_pod(data[:,1:], K)

# save the cluster assignments and centers
cluster_assignments = clustered_data[0]
cluster_centers = clustered_data[1]

classes = cluster_assignments
"""


predictions = np.zeros((movie_count, 2))

predictions[:, 0] = data[:, 0]



predictions[:, 1] = classes

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

        prominent_movie_IDs = movieIds[arr.argsort()[:12]]

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




"""
for n_clusters in range(2,30):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data[:,1:]) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, max_iter=10000, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(data[:,1:])

    cluster_labels = clusterer.fit_predict(data[:,1:])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data[:,1:], cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data[:,1:], cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")


    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()


"""
for i in range(np.shape(genres_distribution)[1]):

    genres_distribution[:, i] /= sum(genres_distribution[:, i])

for i in range(np.shape(genres_distribution)[0]):

    genres_distribution[i,:-1] /= genres_distribution[i,-1]


print("I am happy :)")




"""
#code that draws the elbow plot:

distortions = []
K = range(7, 40)

for cluster_number in K:

    kmeanModel = KMeans(n_clusters=cluster_number, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
    kmeanModel.fit(data)
    distortions.append(kmeanModel.inertia_)


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""

print("hi")

print("hi")

print("hi")

print("hi")