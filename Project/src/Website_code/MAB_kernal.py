import csv
import random

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


def movies_in_cluter(cluster_ID, clustering, number_of_movies, rating_csv):
    """
    :returns finds the IDs of the most prominant movies in a cluster.
    """
    ratings = np.array(rating_csv)
    ratings = ratings[:, 1:]  # drop first column
    ratings = ratings[ratings[:, 0].argsort()]  # sort to first movie Id

    movieIds = clustering[np.where(clustering[:, 1] == cluster_ID)][:, 0]  # movieIdList of 0. cluster

    arr = np.array([])
    for i, ind in enumerate(movieIds):
        arr = np.append(arr, ratings[np.where(ratings[:, 0] == movieIds[i])].sum(axis=0)[1])

    if np.shape(arr)[0] >= number_of_movies:

        prominent_movie_IDs = movieIds[arr.argsort()[::-1][:number_of_movies]]

    else:

        prominent_movie_IDs = movieIds[arr.argsort()]

    return prominent_movie_IDs


def select_movie(movie_IDs, watched_movies, currently_recommended_movies):
    """
    :returns a random random movie from the given list without recommending the a watched movie or recommending the same move twice in one round.
    """
    picked_ID = random.choice(movie_IDs)

    if (picked_ID in watched_movies) or (picked_ID in currently_recommended_movies):

        return select_movie(movie_IDs, watched_movies, currently_recommended_movies)

    else:

        return picked_ID


def movie_names_from_IDs(prominent_movie_IDs, movies):
    """
    :returns list of movie names from list of movie IDs.
    """
    prominent_movie_names = []

    for movie_ID in prominent_movie_IDs:
        index = np.where(movies[:, 0] == movie_ID)

        name = movies[index, 1]

        prominent_movie_names.append(name[0][0])

    return prominent_movie_names


# ***** ****** ***** *****
# these are the parameters of the system.
number_of_recomendations_per_round = 10
number_of_movies_to_consider_in_each_cluster = 60
show_movie_clusters = True
# ***** ***** ***** *****

rating_csv = pd.read_csv('new_rating.csv', header=None)
movie_csv = pd.read_csv('movie.csv')
movies = np.array(movie_csv)

data_csv = pd.read_csv('som_clustering.csv', header=None)
clustering = np.array(data_csv)

ratings = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

K = 9  # number of arms
M = 10  # support size of the distributions

# no need to re-read the ratings file every time as it doesnt change in the demo.
try:
    data_csv = pd.read_csv('movies_in_clusters.csv', header=None)
    movies_in_clusters = np.array(data_csv)

except:
    movies_in_clusters = np.zeros((K, number_of_movies_to_consider_in_each_cluster))

    for i in range(K):
        movies = movies_in_cluter(i, clustering, number_of_movies_to_consider_in_each_cluster, rating_csv)
        movies_in_clusters[i, :len(movies)] = movies

    with open('movies_in_clusters.csv', 'w') as csvfile:

        csv_writer = csv.writer(csvfile)

        csv_writer.writerows(movies_in_clusters)


def fast_movies_in_cluster(cluster_ID):
    """
    :returns IDs of the most prominant movies in a cluster with out reading the ratings file.
    """
    movie_IDs = movies_in_clusters[cluster_ID, :]

    movie_IDs = movie_IDs[movie_IDs != 0]

    return movie_IDs


weights = np.arange(M + 1) / M
weights = weights[1:]

alpha = np.ones((K, M))


def choose_arm_to_sample(alpha):
    L = np.zeros(np.shape(alpha))

    for k in range(K):
        L[k, :] = np.random.dirichlet(alpha[k, :])

    I = L.dot(weights.T)
    It = np.argmax(I)

    return It

watched_movies = []

for t in range(1000):  # 1000 is an arbitery big number.

    # picking the movie cluster to recomend.
    recomendation_clusters = np.zeros(number_of_recomendations_per_round)
    recomendation_IDs = []

    for i in range(number_of_recomendations_per_round):
        recomendation_clusters[i] = choose_arm_to_sample(alpha)

        temp_IDs = fast_movies_in_cluster(int(recomendation_clusters[i]))

        temp_ID = select_movie(temp_IDs, watched_movies, recomendation_IDs)

        recomendation_IDs.append(int(temp_ID))

    # showing the movies to the user

    recomendation_names = movie_names_from_IDs(recomendation_IDs, movies)

    print("\n\nRecomended movies:\n")

    for i in range(len(recomendation_IDs)):

        if show_movie_clusters:

            print(
                str(i + 1) + ".) Cluster " + str(recomendation_clusters[i]) + ": " +
                recomendation_names[i])

        else:

            print(str(i + 1) + ".) " + recomendation_names[i])

    It = int(input("Specify the choosen movie's order in the given list:"))

    while not (It in range(1, number_of_recomendations_per_round + 1)):
        It = int(input(str(It) + " is not a valid order in the given list, specify the choosen movie's order in the given list:"))

    rt = float(input("give a rating (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5):"))

    while not (rt in ratings):
        rt = float(input(str(rt) + " not a valid Id, specify the choosen movies ID number:"))

    # updating these so they are meaningful for the array
    It = int(recomendation_clusters[It - 1])
    rt = int(rt * 2 - 1)

    alpha[It, rt] += 1

print("I am happy")
