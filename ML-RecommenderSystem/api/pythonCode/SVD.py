import pandas as pd
import numpy as np
import os

credits_df = pd.read_csv('./tmdb_5000_credits.csv')
movies_df = pd.read_csv('./tmdb_5000_movies.csv')

credits_df.columns = ['id','tittle','cast','crew']
df = movies_df.merge(credits_df, on='id')

df = df.drop(['budget', 'homepage',
              'original_title', 'production_companies',
              'release_date', 'revenue', 'runtime',
              'spoken_languages', 'status', 'tagline','tittle'], axis=1)

df.columns = ['genres', 'id', 'keywords',
              'language', 'overview',
              'popularity', 'countries',
              'title', 'vote_average',
              'vote_count', 'cast', 'crew']

df['overview'] = df['overview'].fillna('')

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

ratings = pd.read_csv('./api/pythonCode/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], Reader())
# create svd
svd = SVD()

# cross validation
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# fit data
trainset = data.build_full_trainset()
svd.fit(trainset)
#ratings[ratings['userId'] == 1]
userID = os.environ["userID"]
movieID = os.environ["movieID"]

finalValue = svd.predict(int(userID), int(movieID), 3).est

print("test==",finalValue)