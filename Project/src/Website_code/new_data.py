#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import numpy
import pandas as pd


# In[2]:


def create_new_data(user_count, movie_count,ratings):
       
    rows,cols = (movie_count,user_count)
    result = np.zeros((rows,cols))
    row = 0
    current_user = 1
    for user_id in range(1,user_count+1):
        
        while(ratings[row,0] == current_user):
            
            filled_row = int(ratings[row,1]) - 1 #movie_id
            col = user_id - 1
            result[filled_row,col] = ratings[row,2]
            row += 1
            
        current_user += 1     
    return result
    


# In[3]:


def write_csv(new_data):
    
    with open('Desktop/new_data.csv', 'w') as csvfile:
        
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerows(new_data)


# In[4]:



#def main():

rating_csv = pd.read_csv('rating.csv')
rating_csv = rating_csv.drop('timestamp', 1)
ratings = np.array(rating_csv)
#ratings = np.array(ratings[:2000000, :])
ratings[:, [0,1]] = ratings[:, [0,1]].astype(int)
#print(ratings[0,1])
row,column = ratings.shape
#print("Row", row)
#print("Column", column)


#result = remove_users(ratings,200)

#print(result)
#print(result.shape)
    
    
#main()


# In[5]:


#def remove_users(ratings,min_movie_count):
min_user_count = 400
unique, counts = np.unique(ratings[:,0] , return_counts=True)
unique = unique.astype(int)
counts = counts > min_user_count
greater = unique[counts]
print(greater)
print("user count", len(greater))

indices = np.where(np.in1d(ratings[:,0], greater))[0]
print(indices)
#result = ratings[np.where(ratings[:,0] == unique[counts])[0]]

#return result 


# In[6]:


ratings = ratings[indices]


# In[7]:


ratings.shape


# In[8]:


#remove movies
min_movie_count = 500
movie_unique, movie_counts = np.unique(ratings[:,1] , return_counts=True)
#print(movie_unique)
#print(movie_counts)
movie_unique = movie_unique.astype(int)
movie_counts = movie_counts > min_movie_count
movie_greater = movie_unique[movie_counts]
print(movie_greater)

movie_greater = [i for i in movie_greater if i < 27278] # these movies do not have valid IDs

print("movie count", len(movie_greater))

movie_indices = np.where(np.in1d(ratings[:,1], movie_greater))[0]
print(movie_indices)


# In[9]:


ratings = ratings[movie_indices]


# In[10]:





# In[12]:


#write to csv
with open('new_rating.csv', 'w') as csvfile:
        
    csvwriter = csv.writer(csvfile)
        
    csvwriter.writerows(ratings)


movies = pd.read_csv('movie.csv')

# In[11]:






