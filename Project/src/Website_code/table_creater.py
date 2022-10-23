#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import numpy
import pandas as pd


# In[ ]:





# In[2]:


rating_csv = pd.read_csv('new_rating.csv', header=None)
ratings = np.array(rating_csv)



# In[3]:


user_ID, counts = np.unique(ratings[:,0] , return_counts=True)
user_ID = user_ID.astype(int) #user ids
print(user_ID)


# In[4]:


movie_ID, movie_counts = np.unique(ratings[:,1] , return_counts=True)
movie_ID = movie_ID.astype(int)
print(movie_ID)


# In[20]:

user_count = len(user_ID)
movie_count = len(movie_ID)


print("a")


# In[ ]:


rows,cols = (movie_count,user_count)

cols += 1

result = np.zeros((rows,cols))
#result[:,:] = np.NaN # Un comment to generate NaN in place of 0

row = 0
result[:,0] = movie_ID
print(result.astype(int))
print(ratings)



user_index = 0



for i in range(len(ratings)):

    if ratings[i,0] == user_ID[user_index]:

        filled_row = np.where(result[:, 0] == ratings[i, 1])
        result[filled_row, user_index + 1] = ratings[i, 2]

    else: # we moved on to the next user

        user_index += 1

        filled_row = np.where(result[:, 0] == ratings[i, 1])
        result[filled_row, user_index + 1] = ratings[i, 2]


"""

col = 1
row = 0

for user in user_ID:
    while(ratings[row,0] == user):

        filled_row = np.where(result[:,0] == ratings[row,1])
        result[filled_row,col] = ratings[row,2]
        row += 1

      
    #print(result[i])

    col += 1        

    print(user)
"""

print('hi')

"""
for i in range(1, np.shape(result)[1]): #this replaces blanks with means

    user_sum = sum(result[:, i])
    user_nonzero = np.count_nonzero(result[:, i])
    user_mean = user_sum / user_nonzero

    result[:, i] = np.where(result[:, i] == 0, user_mean, result[:, i])
"""

"""
#this replaces blanks with one mean

sum = sum(result)
nonzero = np.count_nonzero(result)
mean = sum / nonzero

for i in range(1, np.shape(result)[1]): #this replaces blanks with means


    result[:, i] = np.where(result[:, i] == 0, mean, result[:, i])

"""

with open('movie_ID_with_ratings_with_one_mean.csv', 'w') as csvfile:

    csv_writer = csv.writer(csvfile)

    csv_writer.writerows(result)

print(result)











