# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:45:47 2023

@author: hamed
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


path = 'C:\\Users\\hamed\\Desktop\\clustering\\wine-clustering.csv'
data = pd.read_csv(path)

scaler = StandardScaler()
data_scaler = scaler.fit_transform(data)
pca = PCA(n_components=2)
wine = pca.fit_transform(data_scaler)
data = pd.DataFrame(wine,columns = ["PCA1","PCA2"])
print(data.describe().T)

clusters = []
for i in range(1,10):
    km = KMeans(n_clusters=i).fit(data)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x= list(range(1,10)),y = clusters , ax = ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
plt.show() 

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(221)
km5 = KMeans(n_clusters=5).fit(data)
data['Labels'] = km5.labels_
sns.scatterplot(x=data['PCA1'],y=data['PCA2'],hue=data['Labels'],style=data['Labels'],
                palette=sns.color_palette('hls', 5), s=60, ax=ax)

ax = fig.add_subplot(222)

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(data)
data['Labels'] = agglom.labels_
sns.scatterplot(x=data['PCA1'],y=data['PCA2'], hue=data['Labels'], style=data['Labels'],
                palette=sns.color_palette('hls', 5), s=60, ax=ax)
ax.set_title('Agglomerative with 5 Clusters')  

ax = fig.add_subplot(223)

db = DBSCAN().fit(data)
data['Labels'] = db.labels_
sns.scatterplot(x=data['PCA1'],y=data['PCA2'], hue=data['Labels'], style=data['Labels'],
                palette=sns.color_palette('hls',  np.unique(data['Labels']).shape[0]),
                s=60, ax=ax)
ax.set_title('DBSCAN with epsilon 11, min samples 6')

ax = fig.add_subplot(224)

bandwidth = estimate_bandwidth(data, quantile=0.1)
ms = MeanShift(bandwidth= bandwidth).fit(data)
data['Labels'] = ms.labels_
sns.scatterplot(x=data['PCA1'],y=data['PCA2'], hue=data['Labels'], 
                palette=sns.color_palette('hls', np.unique(data['Labels']).shape[0]), ax=ax)
ax.set_title('MeanShift')

plt.tight_layout()
plt.show()