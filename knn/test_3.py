#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 下午4:07
# @Author  : Leon
# @Site    : 
# @File    : test_3.py
# @Software: PyCharm
# @Description: 糖尿病预测，样本数据来自：https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from utils.utils import plot_learning_curve


# 1. 加载数据
data = pd.read_csv('diabetes.csv')  # 768 个样本，8个特征，Outcome 为标记值，0 表示没有糖尿病，1表示有糖尿病
print 'diabetes dataset shape {}'.format(data.shape)
print data.head()
print data.groupby('Outcome').size()  # 阳性、阴性的个数

# 分离特征值；把数据集划分为训练集和测试集
X = data.iloc[:, 0:8]
Y = data.iloc[:, 8]
print 'shape of X {}; shape of Y {}'.format(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 2. 模型比较，使用普通的 k-均值算法、带权重的 k-均值算法以及指定半径的 k-均值算法 分别对数据集进行拟合并计算评分
# 构造 3 个模型
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=2, weights='distance')))
models.append(('KNN', RadiusNeighborsClassifier(n_neighbors=2, radius=500)))

# 分别训练 3 个模型，并计算评分
results = []
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
for i in range(len(results)):
    print 'name: {}, score: {}'.format(results[i][0], results[i][1])

# 从上面的结果看，普通的 k-均值算法性能最好。但可能是训练样本和测试样本是随机分配导致的，所以应该多次随机分配训练数据集和交叉验证数据集，
# 然后求模型准确性的平均值，scikit-learn 提供了 KFold 和 cross_val_score() 函数来处理。
print '\n'
results = []
for name, model in models:
    kfold = KFold(n_splits=10)  # 把数据集分为10份，其中一份会作为交叉验证数据集来计算模型准确性，另外9份作为训练数据集
    cv_result = cross_val_score(model, X, Y, cv=kfold)  # 计算10次不同训练数据集和交叉验证数据集组合得到的模型准确性评分
    results.append((name, cv_result))
for i in range(len(results)):
    print 'name: {}; cross val score: {}'.format(results[i][0], results[i][1].mean())
# 结果显示还是普通的 k-均值算法性能最好

print '\n'
# 3. 模型分析
# 使用普通的 k-均值对数据集进行训练，查看对训练样本的拟合情况以及对测试样本的准确性的预测情况
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
train_score = knn.score(X_train, Y_train)
test_score = knn.score(X_test, Y_test)
print 'train score: {}; test score: {}'.format(train_score, test_score)
# train score: 0.822475570033; test score: 0.707792207792

# 拟合情况不佳，原因一：模型太简单，无法很好的拟合训练样本数据；原因二：模型的准确性欠佳
# 画出学习曲线
knn = KNeighborsClassifier(n_neighbors=2)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 6), dpi=200)
plot_learning_curve(plt, knn, 'Learn Curve for KNN Diabetes', X, Y, ylim=(0.1, 1.01), cv=cv)
plt.show()
# 从上图可以看出，训练样本评分较低，切测试样本与训练样本差距较大，这是典型的欠拟合现象。
# k-均值算法没有更好的措施来解决欠拟合问题
