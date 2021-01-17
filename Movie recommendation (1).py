#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[22]:


column_name=['user_id','movie_id','rating','timestamp']
df= pd.read_csv('u.data', sep="\t",names=column_name)
df


# In[23]:


movies_title=pd.read_csv('u.item', sep="\|",header=None,encoding='ISO-8859-1', engine='python')


# In[24]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=['movie_id','title']
movies_titles.head()


# In[25]:


df=pd.merge(df,movies_titles,on="movie_id")
df


# In[26]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings


# In[27]:


ratings['total ratings']= pd.DataFrame(df.groupby('title').count()['rating'])
ratings


# In[28]:


movie=df.pivot_table(index='user_id',columns='title',values="rating")


# In[29]:


def predictor(movie_name):
    movie_user_ratings=movie[movie_name]
    similar=movie.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['total ratings'])
    prediction=corr_movie[corr_movie['total ratings']>100].sort_values('correlation',ascending=False)
    return prediction


# In[30]:


predict_my_movie= predictor('12 Angry Men (1957)')
predict_my_movie.head()


# In[32]:


print("Watch this movie next")
predict_my_movie.head(1)


# In[ ]:




