#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 下午3:24
# @Author  : Leon
# @Site    : 
# @File    : test_1.py
# @Software: PyCharm
# @Description: 使用 k-近邻算法进行分类
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# 1.生成已标记的数据集
centers = [[-2, 2], [2, 2], [0, 4]]
# 使用 make_blobs() 函数生成数据集，分布在以 centers 指定中心点的周围，cluster_std 标准差指定生成的点的分布的松散程度
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.6)
# 画出数据
plt.figure(figsize=(16, 10), dpi=144)
c = np.array(centers)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')  # 画出样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')  # 画出中心点
plt.show()

# 2. 使用 KNeighborsClassifier 算法训练，选择参数 k=5
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# 3. 对一个新的样本进行预测
X_sample = np.array([0, 2]).reshape(-1, 2)
y_sample = clf.predict(X_sample)
# 把样本周围距离最近的 k（5）个点取出来
neighbors = clf.kneighbors(X_sample, return_distance=False)  # 返回是样本的位置集合

# 4. 画出示意图
plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')  # 画出样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')  # 画出中心点
plt.scatter(X_sample[:, 0], X_sample[:, 1], marker="x", c=y_sample, s=100, cmap='cool')  # 待预测的点

# 预测点与距离最近的5个样本的连线
for i in neighbors[0]:
    plt.plot([X[i][0], X_sample[:, 0]], [X[i][1], X_sample[:, 1]], 'k--', linewidth=0.6)
plt.show()
