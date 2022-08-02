import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

header = ['userId','movieId','rating','timestamp']
dataset = pd.read_csv('./api/pythonCode/user.data',sep='\t',names=header)
#print(dataset.head())

#dataset = pd.read_csv('/Users/crystalmccay/Documents/ML_PROJ_1/ML-RecommenderSystem/ratings_small.csv')
#print(dataset.head())

dataset.shape

n_users = dataset.userId.unique().shape[0]
n_items = dataset.movieId.unique().shape[0]
n_items = dataset['movieId'].max()
A = np.zeros((n_users,n_items))
for line in dataset.itertuples():
    A[line[1]-1,line[2]-1] = line[3]
#print("Original rating matrix : ",A)

for i in range(len(A)):
  for j in range(len(A[0])):
    if A[i][j]>=3:
      A[i][j]=1
    else:
      A[i][j]=0

csr_sample = csr_matrix(A)
#print(csr_sample)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
knn.fit(csr_sample)

userID = os.environ["userID"]

dataset_sort_des = dataset.sort_values(['userId', 'timestamp'], ascending=[True, False])
filter1 = dataset_sort_des[dataset_sort_des['userId'] == int(userID)].movieId
filter1 = filter1.tolist()
filter1 = filter1[:20]
#print("Items liked by user: ",filter1)

distances1=[]
indices1=[]
for x in filter1:
  distances , indices = knn.kneighbors(csr_sample[x],n_neighbors=3)
  indices = indices.flatten()
  indices= indices[1:]
  indices1.extend(indices)
print(indices1)





