
import csv
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



movie_csv = pd.read_csv('movie.csv')
movies = np.array(movie_csv)
genres = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir', '(no genres listed)']


for i in range(np.shape(movies)[0]):

    movie_genres = movies[i, 2]

    movie_genres_list = movie_genres.split("|")

    for genre in movie_genres_list:

        if genre not in genres:

            genres.append(genre)


print("I am happy :)")












