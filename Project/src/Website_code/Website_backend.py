import numpy as np
import pandas as pd
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

#cors
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:1337",
    "http://localhost:8080",
    "http://localhost:1337/index.html",
    "http://localhost:1337/page1.html"
]


#
movie_csv = pd.read_csv('movie.csv')
k_means_cluster_csv = pd.read_csv('k-means_clustering.csv')
tag_cluster_csv = pd.read_csv('tag_clustering.csv')
som_cluster_csv = pd.read_csv('som_clustering.csv')
rating_csv = pd.read_csv('new_rating.csv', header=None)
movie = np.array(movie_csv)
k_means_cluster = np.array(k_means_cluster_csv)
tag_cluster = np.array(tag_cluster_csv)
som_cluster = np.array(som_cluster_csv)

def getRandom20():
    first100 = np.loadtxt("your_file.txt",dtype='int')
    #print(arr)
    random20 = []
    for i in range(20):
        random20 = np.append(random20,random.randint(0, 99))
    first100arr = []
    first100arrIds = []
    for i in range(100):
        first100arr = np.append(first100arr,movie[np.where(movie[:, 0] == first100[i])][:,1])
        first100arrIds = np.append(first100arrIds,movie[np.where(movie[:, 0] == first100[i])][:,0])

    randomBest20 = []
    randombest20ids = []
    for i in range(20):
        randomBest20 = np.append(randomBest20,first100arr[int(random20[i])])
        randombest20ids = np.append(randombest20ids,first100arrIds[int(random20[i])])
    return randomBest20,randombest20ids

def getItsClusters20(clustermethod,movie_ID):

    #get cluster ID
    index = np.where(clustermethod[:,0] == movie_ID)

    cluster_ID = clustermethod[index, 1]

    cluster_ID = cluster_ID[0][0]
    ratings = np.array(rating_csv)
    ratings = ratings[:, 1:]  # drop first column
    ratings = ratings[ratings[:, 0].argsort()]  # sort to first movie Id

    movieIds = clustermethod[np.where(clustermethod[:, 1] == cluster_ID)][:, 0]  # movieIdList of 0. cluster


    arr = np.array([])
    for i, ind in enumerate(movieIds):
        arr = np.append(arr, ratings[np.where(ratings[:, 0] == movieIds[i])].sum(axis=0)[1])

    if np.shape(arr)[0] >= 24:

        prominent_movie_IDs = movieIds[arr.argsort()[::-1][:24]]

    else:

        prominent_movie_IDs = movieIds[arr.argsort()]

    prominent_movie_names = []

    for movie_ID in prominent_movie_IDs:

        index = np.where(movie[:,0] == movie_ID)

        name = movie[index, 1]

        prominent_movie_names.append(name[0])


    return prominent_movie_names

app = FastAPI()
##app rcors
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#pulp fiction id 296
class Film(BaseModel):
    id:int
    title: str


class FilmList(BaseModel):
    items: List[Film]
    
class FilmOnlyTitle(BaseModel):
    title: str


class FilmListTitle(BaseModel):
    items: List[FilmOnlyTitle]
    



@app.get("/getRandom20")
async def root():
    films20,films20ids = getRandom20()
    data=[]
    for i in range(20):
        item = {"id": films20ids[i]}
        item["title"] = films20[i]
        data.append(item)

    films = {"items": data}
    print(films)
    return films

@app.get("/getClusterKmeans")
async def read_item(id:int):
    films20 = getItsClusters20(k_means_cluster,id)
    a = films20[0][0]
    print(a)
    data=[]
    for i,ind in enumerate(films20):
        item = {"title": films20[i][0]}
        data.append(item)

    return data

@app.get("/getClusterTag")
async def read_item(id:int):
    films20 = getItsClusters20(tag_cluster,id)
    a = films20[0][0]
    print(a)
    data=[]
    for i,ind in enumerate(films20):
        item = {"title": films20[i][0]}
        data.append(item)

    return data

@app.get("/getClusterSom")
async def read_item(id:int):
    films20 = getItsClusters20(som_cluster,id)
    a = films20[0][0]
    print(a)
    data=[]
    for i,ind in enumerate(films20):
        item = {"title": films20[i][0]}
        data.append(item)

    return data

#import MAB_kernal
import csv
import random

import numpy as np
import numpy
from numpy.core.records import array
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
recomendation_clusters = []
def getRecommended():  # 1000 is an arbitery big number.
    global recomendation_clusters
    recomendation_clusters = np.zeros(number_of_recomendations_per_round)
    recomendation_IDs = []


    # picking the movie cluster to recomend.

    for i in range(number_of_recomendations_per_round):
        recomendation_clusters[i] = choose_arm_to_sample(alpha)

        temp_IDs = fast_movies_in_cluster(int(recomendation_clusters[i]))

        temp_ID = select_movie(temp_IDs, watched_movies, recomendation_IDs)

        recomendation_IDs.append(int(temp_ID))

    # showing the movies to the user

    recomendation_names = movie_names_from_IDs(recomendation_IDs, movies)

    print("\n\nRecomended movies:\n")
    arrayFilm = []
    ids = []
    
    for i in range(len(recomendation_IDs)):

        if show_movie_clusters:
            arrayFilm = np.append(arrayFilm,(
                str(i + 1) + '.) Cluster ' + str(recomendation_clusters[i]) + ': ' +
                recomendation_names[i])
            )
        else:
            arrayFilm = np.append(arrayFilm,
            (str(i + 1) + '.) ' + recomendation_names[i])
            )

    return arrayFilm


def updateArray(movieOrder,ratingS):


    # picking the movie cluster to recomend.



    It = movieOrder#int(input("Specify the choosen movie's order in the given list:"))

#    while not (It in range(1, number_of_recomendations_per_round + 1)):
 #       It = int(input(str(It) + " is not a valid order in the given list, specify the choosen movie's order in the given list:"))

    rt = ratingS#float(input("give a rating (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5):"))

  #  while not (rt in ratings):
   #     rt = float(input(str(rt) + " not a valid Id, specify the choosen movies ID number:"))

    # updating these so they are meaningful for the array
   
    It = int(recomendation_clusters[It - 1])
    rt = int(rt * 2 - 1)

    alpha[It, rt] += 1
    print(It)
    print("reco")
    print(recomendation_clusters)
    print(alpha)
    

print("I am happy")

#get mab
@app.get("/mabGet")
async def read_item():
    film = getRecommended()
    data =[]
    for i,ind in enumerate(film):
        print(i)
        item = {"id": (i+1)}
        item["title"] = film[i]
        data.append(item)

    films = {"items": data}
    return films
class Item(BaseModel):
    movieOrder: int
    ratingS: float
@app.post("/mabUpdate")
async def change_item(item: Item):
    updateArray(item.movieOrder,item.ratingS)
    return item



