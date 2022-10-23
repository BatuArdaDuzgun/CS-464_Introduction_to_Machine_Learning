#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn_som.som import SOM
import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv('movie_ID_with_ratings_with_mean.csv')
initial_data = dataset
dataset = dataset.iloc[: , 1:]


dataset= np.array(dataset)
initial_data= np.array(initial_data)



som = SOM(m=9, n=1, dim=10638)


# In[6]:


som.fit(dataset)


# In[7]:


predictions = som.predict(dataset)



size = (2825,2)
new_data = np.zeros(size)


# In[30]:


predictions = np.array(predictions)



# In[31]:


new_data[:,0] = initial_data[:,0]
new_data[:,1] = predictions


# In[32]:


# In[35]:


import csv


# In[36]:


def write_csv(new_data):
    
    with open('som_clustering.csv', 'w') as csvfile:
        
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerows(new_data)


write_csv(new_data)


# In[ ]:




