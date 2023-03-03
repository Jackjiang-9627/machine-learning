import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
class KNN:
    """实现加权分类，封装成KNN类(kd树)，实现fit，predict，score方法"""
    def __init__(self, k, is_weight, with_kd_tree=True):
        """
        :param k: 近邻点数
        """
        self.kd_tree = None
        self.train_y = None
        self.train_x = None
        self.k_neighbors = k
        self.is_weight = is_weight
        self.with_kd_tree = with_kd_tree

    def fit(self, train_x, train_y):
        """
        KNN算法训练模型即为存储训练集数据，已在构造函数中储存,此处还可定义其它存储方法，比如kd树
        """
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        if self.with_kd_tree:
            self.kd_tree = KDTree(self.train_x, leaf_size=10, metric='minkowski')

    def cal_dis(self, X, y):
        """
        计算距离
        :param X: 训练集数据点
        :param y: 测试集向量（单个测试点）
        :return: 单个测试点与训练集的距离(len(x),)
        """
        return np.sum((X - y) ** 2, axis=1) ** 0.5

    def get_k_neighbors(self, test_x):
        """
        获取前k个临近点
        :return: 前k个临近点的索引位置和标签
        """
        labels = []
        distance = []
        if self.with_kd_tree:
            # 返回对应最近的k个样本的下标，如果return_distance=True同时也返回距离
            distance, index = self.kd_tree.query(test_x, k=self.k_neighbors, return_distance=True)
            # 获取对应最近k个样本的距离
            for i in index:
                labels.append(self.train_y[i])
            return distance, labels
        else:
            for test in test_x:
                dis = self.cal_dis(self.train_x, test)
                dis_df = pd.Series(dis, index=[i for i in range(len(dis))])
                # 排序
                dis_df.sort_values(inplace=True)
                distance.append(dis_df[:self.k_neighbors])
                # 前k个邻近数据点的索引
                dis_k_id = dis_df[:self.k_neighbors].index
                # 前k个标签
                k_labels = [self.train_y[i] for i in dis_k_id]
                labels.append(k_labels)
            return distance, labels

    def predict(self, x):
        """
        多数表决
        is_weight 是否采用加权投
        return X对应的预测标签列表
        """
        labels = pd.DataFrame(self.get_k_neighbors(x)[1])

        if not self.is_weight:
            # 取每行标签中出现次数最多的那个标签
            labels['counts'] = labels.apply(lambda x: x.value_counts().idxmax(), axis=1)
            print(labels)
            return labels['counts'].values
        else:
            # 距离的倒数
            distance_reciprocal = 1 / (np.array(self.get_k_neighbors(x)[0]) + 0.0001)
            # 权重和
            weights_sum = np.sum(distance_reciprocal, axis=1, keepdims=True)
            # 权重矩阵，[weight1, weight2, weight3]
            weights = distance_reciprocal / weights_sum
            pred = []
            for i in range(len(labels)):
                # 遍历待测样本, [label1, label2, label3]
                x = labels.iloc[i]
                # 遍历权重矩阵，每行为待测样本的权重向量[w1, w2, w3]
                w = weights[i]
                df = pd.DataFrame({'labels': x, 'weights': w})
                # 将[label1, label2, label3] 与 [w1, w2, w3] 按label分组相加并取大的值的标签
                df2 = df.groupby('labels').sum()
                # df2.idxmax()是Series对象，索引‘weights’值，即标签
                pred.append(df2.idxmax()['weights'])
            labels['pred'] = pred
            # print(labels)
            return labels['pred'].values

    def score(self, x, y):
        """准确率"""
        y_hat = self.predict(x)
        acc = np.mean(y_hat == y)
        return acc
