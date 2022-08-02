import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
print("Py userId--------")

import pandas as pd
import matplotlib.pyplot as plt
import surprise as p
print(p.__version__)
#from surprise import Dataset
from surprise import Dataset, SVD, evaluate
from surprise import Reader

reader = Reader()

ratings = pd.read_csv('./ratings_small.csv')
ratings.head()


#userId movieId rating timestamp
ratings.columns = ['userId', 'movieId', 'rating','timestamp']

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)

svd = SVD()
#evaluate(svd, data, measures=['RMSE', 'MAE'])

trainset = data.build_full_trainset()
svd.fit(trainset)

ratings[ratings['userId'] == 1]

userId = "1"
movieId = "302"
#userId = os.environ["USERID"]
#movieId = os.environ["MOVIEID"]

print("Py userId", userId)
print("Py movieId", movieId)

def get_pred_rating(userId,movieId):
    pred_rating = svd.predict(userId, movieId, 3)
    return pred_rating

get_pred_rating(userId,movieId)