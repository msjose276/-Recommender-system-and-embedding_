import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv('./tmdb_5000_credits.csv')
df2=pd.read_csv('./tmdb_5000_movies.csv')

#df2 = df2.rename(columns={"original_title":"title"}, inplace=True)

C= df2['vote_average'].mean()
m= df2['vote_count'].quantile(0.9)
q_movies = df2.copy().loc[df2['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

pop= q_movies.sort_values('popularity', ascending=False)

new = pop.filter(['id','original_language','original_title'], axis=1)
new.rename(columns={"original_title":"title"}, inplace=True)

new = new[:10]

print(new)
new.to_csv(r'./api/MLResult/result.csv', index=False)