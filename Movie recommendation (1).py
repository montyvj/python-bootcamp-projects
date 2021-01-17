import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

column_name=['user_id','movie_id','rating','timestamp']
df= pd.read_csv('u.data', sep="\t",names=column_name)
df
movies_title=pd.read_csv('u.item', sep="\|",header=None,encoding='ISO-8859-1', engine='python')
movies_titles=movies_title[[0,1]]
movies_titles.columns=['movie_id','title']
movies_titles.head()
df=pd.merge(df,movies_titles,on="movie_id")
df
ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings
ratings['total ratings']= pd.DataFrame(df.groupby('title').count()['rating'])
ratings
movie=df.pivot_table(index='user_id',columns='title',values="rating")
def predictor(movie_name):
    movie_user_ratings=movie[movie_name]
    similar=movie.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['total ratings'])
    prediction=corr_movie[corr_movie['total ratings']>100].sort_values('correlation',ascending=False)
    return prediction
predict_my_movie= predictor('12 Angry Men (1957)')
predict_my_movie.head()
print("Watch this movie next")
predict_my_movie.head(1)



