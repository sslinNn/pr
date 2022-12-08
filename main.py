import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# 1
dataset = pd.read_csv('weight-height.csv')
""" Уникальный Рост """
uniqHeight = dataset['Height'].unique()
# print(len(uniqHeight))

""" Уникальный Вес """
uniqWeight = dataset['Weight'].unique()
# print(len(uniqWeight))

# 2
dataset['Height'] = dataset['Height'] * 2.54
dataset['Weight'] = dataset['Weight'] * 0.45359237
# print(dataset.head(10))


# 3
Male_df = pd.DataFrame(dataset.loc[dataset['Gender'] == 'Male'])        #Делит на мужчин
# print(Male_df.head())
Female_df = pd.DataFrame(dataset.loc[dataset['Gender'] == 'Female'])        #Делит на женщин
# print(Female_df.head())

# Функция для нахождения Мат.Ожидания
def MathExp(height = False, weight = False):
    if height:
        MathE = (dataset['Height'].sum()) / (dataset['Height'].count())
        return f'Мат.Ожидание для "Height" - {MathE}'
    if weight:
        MathE = (dataset['Weight'].sum()) / (dataset['Weight'].count())
        return f'Мат.Ожидание для "Weight" - {MathE}'
# print(MathExp(height=True))
# print(MathExp(weight=True))

# Функция для нахождения STD
def Std(height = False, weight = False):
    if height:
        std = (dataset['Height']).std()
        return f'Стандартное отклонение для "Height" - {std}'
    if weight:
        std = (dataset['Weight']).std()
        return f'Стандартное отклонение для "Weight" - {std}'
# print(Std(height=True))
# print(Std(weight=True))

# Функция для нахождения min & max
def MinMax(height = False, weight = False):
    if height:
        max1 = (dataset['Height']).max()
        min1 = (dataset['Height']).min()
        return f'Max "Height" - {max1}' \
               f'Min "Height" - {min1}'
    if weight:
        max1 = (dataset['Weight']).max()
        min1 = (dataset['Weight']).min()
        return f'Max "Weight" - {max1} ' \
               f'Min "Weight" - {min1}'
# print(MinMax(height=True))
# print(MinMax(weight=True))

# 4
"""Man's"""
def Mplot():
    Male_df.plot(xlabel ='Range', y = ['Height', 'Weight'])
    plt.show()
# Mplot()

"""Woman's"""
def Fplot():
    Female_df.plot(xlabel ='Range', y = ['Height', 'Weight'])
    plt.show()
# Fplot()

# 6
def HierarchicalClustering():
    X = dataset.iloc[:, [1, 2]].values
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.legend()
    plt.show()
# HierarchicalClustering()

def KMeansClustering():
    wcss = []
    X = dataset.iloc[:, [1, 2]].values
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.legend()
    plt.show()
# KMeansClustering()