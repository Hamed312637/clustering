# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:09:01 2023

@author: hamed
"""

from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



path = 'C:\\Users\\hamed\\Desktop\\clustering\\Mall_Customers.csv'
data = pd.read_csv(path)
data.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)

X = data.drop(['CustomerID', 'Gender'],axis=1)
sns.pairplot(data.drop('CustomerID',axis=1),hue='Gender',aspect=1.5)
plt.show()


clusters =[]
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
    

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters,ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
ax.annotate('Possible Elbow Point',xy=(3,140000),xytext=(3,50000),xycoords='data',
            arrowprops= dict(arrowstyle= '->', connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
plt.show() 


km3 = KMeans(n_clusters=3).fit(X)
X['Labels']=km3.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x=X['Income'],y=X['Score'],hue=X['Labels'],
                palette=sns.color_palette('hls',3)) 
plt.title('KMeans with 3 Clusters')
plt.show() 

km5 = KMeans(n_clusters=5).fit(X)
X['Labels'] = km5.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x=X['Income'],y=X['Score'],hue=X['Labels'],
                palette=sns.color_palette('hls',5)) 
plt.show()


from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=5,linkage='average').fit(X)
X['Labels'] = agg.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x=X['Income'],y=X['Score'],hue=X['Labels'],
                palette=sns.color_palette('hls',5)) 
ax.set_title('Agglomerative with 5 Clusters')
plt.show()

from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

dist = distance_matrix(X, X)
Z = hierarchy.linkage(dist, 'complete')
plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')

from sklearn.cluster import DBSCAN 

db = DBSCAN(eps=11,min_samples=6).fit(X)
X['labels'] = db.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x=X['Income'],y=X['Score'],hue=X['Labels'],
                palette=sns.color_palette('hls',np.unique(db.labels_).shape[0]))
plt.title('DBSCAN with epsilon 11, min samples 6')
plt.show()


from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(X,quantile=0.1)
ms = MeanShift(bandwidth=bandwidth).fit(X)

X['Labels'] = ms.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X['Income'],y= X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', np.unique(ms.labels_).shape[0]))

plt.title('MeanShift')
plt.show()


