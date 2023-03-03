"""简单实现一下等权分类，封装成KNN类，实现fit，predict，score方法"""
import numpy as np
import pandas as pd


class KNN:
    def __init__(self, k):
        """
        :param k: 近邻点数
        """
        self.k_neighbors = k

    def fit(self, train_x, train_y):
        """
        KNN算法训练模型即为存储训练集数据，已在构造函数中储存
        此处还可定义其它存储方法，比如kd树
        """
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)

    def get_k_neighbors(self, test_x):
        """
        获取前k个临近点
        :return: 前k个临近点的索引位置和标签
        """

        # 计算距离
        def cal_dis(X, y):
            return np.sum((X - y) ** 2, axis=1) ** 0.5

        labels = []
        for test in test_x:
            dis = cal_dis(self.train_x, test)
            dis_df = pd.Series(dis, index=[i for i in range(len(dis))])
            # 排序
            dis_df.sort_values(inplace=True)
            # 前k个邻近数据点的索引
            dis_k_id = dis_df[:self.k_neighbors].index
            # 前k个标签
            k_labels = [self.train_y[i] for i in dis_k_id]
            labels.append(k_labels)
        return labels

    def predict(self, x, is_weight=False):
        """
        多数表决
        is_weight 是否采用加权投
        return X对应的预测标签列表
        """
        labels = pd.DataFrame(self.get_k_neighbors(x))
        # labels['Row_sum'] = labels.apply(lambda x: x.sum(), axis=1)  # 按行求和，添加为新列
        # labels['Pred'] = labels.apply(lambda x: 1 if x['Row_sum'] > 0 else -1, axis=1)
        labels['Pred'] = labels.apply(lambda x: x.value_counts().idxmax(), axis=1)
        # print(labels)
        return labels['Pred'].values

    def score(self, x, y):
        """准确率"""
        y_hat = self.predict(x)
        acc = np.mean(y_hat == y)
        return acc


if __name__ == '__main__':
    # 训练数据
    T = [[3, 104, -1], [98, 2, 1], [2, 100, -1],
         [101, 10, 1], [99, 5, 1], [1, 81, -1]]
    T = np.array(T)
    train_x, train_y = (T[:, :-1], T[:, -1])
    # 预测数据，判断其电影类型
    test_x = [[18, 90], [50, 50]]
    knn = KNN(3)
    # 训练
    knn.fit(train_x, train_y)
    # 预测
    print('预测结果：{}'.format(knn.predict(test_x)))
    # # 准确率
    print('预测结果：{}'.format(knn.predict(train_x)))
    print('预测准确率为：{}'.format(knn.score(train_x, train_y)))

    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    knn2 = KNN(k=3)
    knn2.fit(x_train, y_train)
    print('测试集预测结果：{}'.format(knn2.predict(x_test)))
    print('测试集预测准确率为：{}'.format(knn2.score(x_test, y_test)))
    print('训练集预测结果：{}'.format(knn2.predict(x_train)))
    print('预训练集测准确率为：{}'.format(knn2.score(x_train, y_train)))