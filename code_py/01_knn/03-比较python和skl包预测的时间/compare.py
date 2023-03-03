import sys
sys.path.append('../01-基于python实现的电影数据分类/04_KNN加权分类封装/')
from KNN import KNN
import time

"""python测试鸢尾花数据所花时间"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, Y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 开始计时
since = time.time()
knn1 = KNN(k=3, is_weight=True, with_kd_tree=True)
knn1.fit(x_train, y_train)
pred = knn1.predict(x_test)
acc = knn1.score(x_test, y_test)
# 预测结束，停止计时
time_elapsed = time.time() - since
print(f'python预测用时:{time_elapsed:.4f}s')
# python预测用时:0.0626s
"""python测试鸢尾花数据所花时间"""
from sklearn import neighbors
# 开始计时
since = time.time()
knn2 = neighbors.KNeighborsClassifier(n_neighbors=3, leaf_size=10, algorithm='kd_tree', weights='distance')
knn2.fit(x_train, y_train)
pred1 = knn2.predict(x_test)
acc1 = knn2.score(x_test, y_test)
# 预测结束，停止计时
time_elapsed = time.time() - since
print(f'skl预测用时:{time_elapsed}s')
# skl预测用时:0.0s
