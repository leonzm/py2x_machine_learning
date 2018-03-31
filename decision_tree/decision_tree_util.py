#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/8 上午11:01
# @Author  : Leon
# @Site    : 
# @File    : decision_tree_util.py
# @Software: PyCharm
# @Description: 参考别人源码，整理、优化、封装
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plot

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def calc_ent(data_set):
    """
    计算数据集（样本集合）的信息熵（以2为底数）
    :param data_set: 二维 list 的数据集
    :return:
    """
    sample_total = len(data_set)
    sample_label_counts = {}  # {sample_label: count}
    for vector in data_set:
        sample_label = vector[-1]  # 每行的最后一位即为样本的分类标签
        if sample_label not in sample_label_counts:
            sample_label_counts[sample_label] = 0
        sample_label_counts[sample_label] += 1
    ent = 0.0
    for sample_label in sample_label_counts:
        count = sample_label_counts[sample_label]
        prob = 1.0 * count / sample_total
        ent += (- prob * log(prob, 2))
    return ent


def split_data_set(data_set, axis, value):
    """
    对离散变量划分数据集，取出 axis 列 特征值为 value 的所有样本
    :param data_set: 二维 list 的数据集
    :param axis: int 划分所在的列
    :param value: 划分的决策值
    :return:
    """
    sub_data_set = []
    for vector in data_set:
        if vector[axis] == value:
            sub_data_set.append(vector[:axis] + vector[axis + 1:])  # 注意子数据集不包括 axis 列的数据
    return sub_data_set


def split_continuous_data_set(data_set, axis, value, direction):
    """
    对连续变量划分数据集
    :param data_set: 二维 list 的数据集
    :param axis: int 划分所在的列
    :param value: 划分的决策值
    :param direction: 0，划分出大于 value 的数据集；1，划分出小于等于 value 的数据集
    :return:
    """
    sub_data_set = []
    for vector in data_set:
        if direction == 0:
            if vector[axis] > value:
                sub_data_set.append(vector[:axis] + vector[axis + 1:])  # 注意子数据集不包括 axis 列的数据
        else:
            if vector[axis] <= value:
                sub_data_set.append(vector[:axis] + vector[axis + 1:])  # 注意子数据集不包括 axis 列的数据
    return sub_data_set


def choose_best_feature_to_split(data_set, columns):
    """
    选择最好的数据集划分方式
    :param data_set: 二维 list 的数据集
    :param columns: list 二维 list 每列对应的标签
    :return:
    """
    attributes_total = len(data_set[0]) - 1  # 除最后一列是分类标签外，其它都可划分
    base_ent = calc_ent(data_set)  # 信息熵
    best_info_gain = 0.0  # 信息增益
    best_feature = -1  # 划分方式
    split_dict = {}  # 划分集合，{label, split_data}
    for attributes_index in range(attributes_total):
        feature_list = [vector[attributes_index] for vector in data_set]
        # 对连续型特征进行处理
        if type(feature_list[0]).__name__ == 'float' or type(feature_list[0]).__name__ == 'int':
            # 产生 n-1 个候选划分点
            sort_feature_list = sorted(feature_list)
            split_list = []
            for split_index in range(len(sort_feature_list) - 1):
                split_list.append((sort_feature_list[split_index] + sort_feature_list[split_index + 1]) / 2.0)

            best_split_condition_ent = 10000  # 条件熵
            # 求用第 split_index 个候选划分点分时，得到的条件熵（越大越好），并记录最佳划分点
            for split_index in range(len(split_list)):
                value = split_list[split_index]
                # 划分的两个子集
                sub_data_set_0 = split_continuous_data_set(data_set, attributes_index, value, 0)
                sub_data_set_1 = split_continuous_data_set(data_set, attributes_index, value, 1)
                # 两个子集分别占原样本（集合）的比例
                prob0 = 1.0 * len(sub_data_set_0) / len(data_set)
                prob1 = 1.0 * len(sub_data_set_1) / len(data_set)
                # 计算条件熵
                condition_ent = prob0 * calc_ent(sub_data_set_0) + prob1 * calc_ent(sub_data_set_1)

                if condition_ent < best_split_condition_ent:
                    best_split_condition_ent = condition_ent
                    best_split_index = split_index
            # 记录每列的最佳划分值
            split_dict[columns[attributes_index]] = split_list[best_split_index]
            # 信息增益 = 信息熵 - 条件熵
            info_gain = base_ent - best_split_condition_ent

        # 对离散型特征进行处理
        else:
            unique_values = set(feature_list)
            # 计算条件熵
            condition_ent = 0.0
            for value in unique_values:
                sub_data_set = split_data_set(data_set, attributes_index, value)
                prob = 1.0 * len(sub_data_set) / len(data_set)
                condition_ent += prob * calc_ent(sub_data_set)
            info_gain = base_ent - condition_ent

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = attributes_index

    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理，即是否小于等于 best_split_value
    if type(data_set[0][best_feature]).__name__ == 'float' or type(data_set[0][best_feature]).__name__ == 'int':
        best_split_value = split_dict[columns[best_feature]]
        columns[best_feature] = columns[best_feature] + '<=' + str(best_split_value)
        for row_index in range(np.shape(data_set)[0]):
            if data_set[row_index][best_feature] <= best_split_value:
                data_set[row_index][best_feature] = 1
            else:
                data_set[row_index][best_feature] = 0

    return best_feature


def majority_count(class_list):
    """
    特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
    返回类别最多的类别
    :param class_list: 类别集合
    :return:
    """
    class_counts = {}
    for clazz in class_list:
        if clazz not in class_counts:
            class_counts[clazz] = 0
        class_counts[clazz] += 1
    max_class_type = None
    max_class_count = 0
    for clazz in class_counts.iterkeys():
        count = class_counts[clazz]
        if count > max_class_count:
            max_class_type = clazz
            max_class_count = count
    return max_class_type


def create_tree(data_set, labels, data_full, labels_full):
    """
    递归创建决策树
    :param data_set: 当前创建的数据集
    :param labels: 当前创建数据集的标签
    :param data_full: 数据集
    :param labels_full: 数据集的标签
    :return:
    """
    class_list = [example[-1] for example in data_set]  # 类别集合
    if class_list.count(class_list[0]) == len(class_list):  # 只剩一个类别
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_count(class_list)

    # 计算最好的数据集划分方式
    best_feature = choose_best_feature_to_split(data_set, labels)
    best_feature_label = labels[best_feature]

    my_tree = {best_feature_label: {}}
    feature_values = [example[best_feature] for example in data_set]  # 划分列的值的集合
    unique_values = set(feature_values)  # 去重
    if type(data_set[0][best_feature]).__name__ == 'str':
        current_label = labels_full.index(labels[best_feature])
        feature_values_full = [example[current_label] for example in data_full]
        unique_values_full = set(feature_values_full)
    del (labels[best_feature])

    # 针对 best_feature 的每个取值，划分出一个子树
    for value in unique_values:
        sub_labels = labels[:]
        if type(data_set[0][best_feature]).__name__ == 'str':
            unique_values_full.remove(value)
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value),
                                                         sub_labels, data_full, labels_full)

    if type(data_set[0][best_feature]).__name__ == 'str':
        for value in unique_values_full:
            my_tree[best_feature_label][value] = majority_count(class_list)
    return my_tree


def gen_leaf_count(my_tree):
    """
    递归计算叶子节点的数量
    :param my_tree: 树
    :return:
    """
    leaf_count = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leaf_count += gen_leaf_count(second_dict[key])
        else:
            leaf_count += 1
    return leaf_count


def gen_tree_depth(my_tree):
    """
    递归计算树的深度
    :param my_tree: 树
    :return:
    """
    max_depth = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            current_depth = 1 + gen_tree_depth(second_dict[key])
        else:
            current_depth = 1
        if current_depth > max_depth:
            max_depth = current_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    画节点
    :param node_txt:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(current_pt, parent_pt, txt_str):
    """
    画箭头上的文字
    :param current_pt:
    :param parent_pt:
    :param txt_str:
    :return:
    """
    lens = len(txt_str)
    x_mid = (parent_pt[0] + current_pt[0]) / 2.0 - lens * 0.002
    y_mid = (parent_pt[1] + current_pt[1]) / 2.0
    create_plot.ax1.text(x_mid, y_mid, txt_str)


def plot_tree(tree, parent_pt, node_txt):
    """

    :param tree:
    :param parent_pt:
    :param node_txt:
    :return:
    """
    leaf_count = gen_leaf_count(tree)
    first_str = tree.keys()[0]
    cntrPt = (plot_tree.x0ff + (1.0 + float(leaf_count)) / 2.0 / plot_tree.totalW, plot_tree.y0ff)
    plot_mid_text(cntrPt, parent_pt, node_txt)
    plot_node(first_str, cntrPt, parent_pt, decision_node)
    secondDict = tree[first_str]
    plot_tree.y0ff = plot_tree.y0ff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrPt, str(key))
        else:
            plot_tree.x0ff = plot_tree.x0ff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.x0ff, plot_tree.y0ff), cntrPt, leaf_node)
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), cntrPt, str(key))
    plot_tree.y0ff = plot_tree.y0ff + 1.0 / plot_tree.totalD


def create_plot(tree):
    """
    :param tree:
    :return:
    """
    fig = plot.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plot.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(gen_leaf_count(tree))
    plot_tree.totalD = float(gen_tree_depth(tree))
    plot_tree.x0ff = -0.5 / plot_tree.totalW
    plot_tree.y0ff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plot.show()


if __name__ == '__main__':
    df = pd.read_csv('decision_tree_data_english.csv')
    data = df.values[:, 1:].tolist()
    data_full = data[:]
    labels = df.columns.values[1:-1].tolist()
    labels_full = labels[:]
    tree = create_tree(data, labels, data_full, labels_full)
    print tree
    create_plot(tree)
