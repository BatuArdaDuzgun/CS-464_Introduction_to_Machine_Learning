#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:54:58 2021

@author: kckasan
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



dpd = pd.read_csv("genome_scores.csv")
d = dpd.to_numpy()

table = pd.pivot_table(dpd, values='relevance', index=['movieId'],
                    columns=['tagId'], aggfunc=np.sum)



kmeans = KMeans(n_clusters=6, init='k-means++', 
                n_init=10, max_iter=300, tol=0.0001, verbose=0, 
                random_state=None, copy_x=True, algorithm='auto').fit(table)

label = kmeans.labels_

ptable = pd.DataFrame({"movieId":table.index, "label":label})

som = pd.read_csv("som_clustering.csv").astype("int32")
som.columns = ["movieId", "label"]


tfinal = pd.merge(ptable, som, on = "movieId")
tfinal = tfinal.iloc[:,:2]
tfinal.columns = ["movieId", "label"]

tfinal.to_csv("tag_clustering.csv", index = False, header = False)



#Silhoutte Plot

x = np.arange(1,15)

# =============================================================================
# slharr = []
# 
# for i in x:
#     
#     kmeansloop = KMeans(n_clusters=i, init='k-means++', 
#                     n_init=10, max_iter=300, tol=0.0001, verbose=0, 
#                     random_state=None, copy_x=True, algorithm='auto').fit(table)
#     lbl = kmeansloop.labels_
#     sl = silhouette_score(table, lbl)
#     slharr.append(sl)
# 
# plt.clf()
# plt.figure(0)
# plt.plot(x, slharr, "bo-")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.main("Silhouette Plot")
# 
# =============================================================================
    

#Elbow Plot

# =============================================================================
# dst = []
# for i in x:
#     kmeansd = KMeans(n_clusters=i, init='k-means++', 
#         n_init=10, max_iter=300, tol=0.0001, verbose=0, 
#         random_state=None, copy_x=True, algorithm='auto').fit(table)
#     dst.append(kmeansd.inertia_)  
# 
# plt.figure(1)
# plt.plot(x, dst, 'bo-')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow Plot')
# =============================================================================




