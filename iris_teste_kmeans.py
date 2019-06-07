# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:11:30 2019

@author: alef1
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
iris = pd.read_csv("iris.csv")

print(iris.head())

X = iris.iloc[:, 0:4].values


wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'random')

kmeans.fit(X)

print(kmeans.cluster_centers_)


distance = kmeans.fit_transform(X)

print(distance)

labels = kmeans.labels_

print(labels)

