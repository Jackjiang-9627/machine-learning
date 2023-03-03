import numpy as np

from KNN import KNN
# 训练数据
T = [[3, 104, -1], [98, 2, 1], [2, 100, -1],
     [101, 10, 1], [99, 5, 1], [1, 81, -1]]
T = np.array(T)
train_x, train_y = (T[:, :-1], T[:, -1])
# 预测数据，判断其电影类型
test_x = [[18, 90], [50, 50]]
knn = KNN(3, is_weight=True, with_kd_tree=True)
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
knn2 = KNN(k=3, is_weight=True, with_kd_tree=True)
knn2.fit(x_train, y_train)
print('测试集预测结果：{}'.format(knn2.predict(x_test)))
print('测试集预测准确率为：{}'.format(knn2.score(x_test, y_test)))
print('训练集预测结果：{}'.format(knn2.predict(x_train)))
print('预训练集测准确率为：{}'.format(knn2.score(x_train, y_train)))