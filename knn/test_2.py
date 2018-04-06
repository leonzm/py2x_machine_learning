#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 下午3:26
# @Author  : Leon
# @Site    : 
# @File    : test_2.py
# @Software: PyCharm
# @Description: 使用 k-近邻算法进行回归拟合
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# 1. 生成数据集
n_dots = 40
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).ravel()
y += 0.2 * np.random.rand(n_dots) - 0.1  # 添加一些噪声

# 2. 使用 KNeighborsRegressor 来训练模型
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)  # 相比 KNeighborsClassifier 的 y 值，这里的 y 值不是分类而是 f(x)。所以这里的 x 不是离散而是连续的
print knn.score(X, y)  # 计算拟合曲线对训练样本的拟合准确性

# 3. 回归拟合，生成足够密集的点进行预测
T = np.linspace(0, 5, 500)[:, np.newaxis]  # 效果同 reshape(-1, 2)
y_pred = knn.predict(T)

# 4. 画出拟合曲线
plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X, y, c='g', label='data', s=100)  # 训练样本
plt.plot(T, y_pred, c='k', label='prediction', lw=4)  # 拟合曲线
plt.axis('tight')
plt.title('KNeighborsRegressor (k = %i)' % k)
plt.show()
