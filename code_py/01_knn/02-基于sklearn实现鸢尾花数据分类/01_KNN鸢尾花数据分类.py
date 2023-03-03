import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import joblib
# 1、数据加载
names = ['x1', 'x2', 'x3', 'x4', 'y']
data = pd.read_csv('../datas/iris.data', sep=',', names=names)
# 概览数据
print(data.head())
print(data.shape)
print(data["y"].value_counts())  # 分为三类，每类50个样本
# 2、数据进行清洗
# NOTE: 不需要做数据处理
flower = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
data['y'] = data['y'].apply(lambda row: flower[row] if row in flower else 0)
print(data)
data.info()
data['y'] = data['y'].astype(np.int32)
print(data['y'].value_counts())

# 3、获取我们的数据的特征属性X和目标属性Y
X, Y = (data.iloc[:, :-1], data.iloc[:, -1])
print(X.head())

# 4、划分训练集与测试集
# train_size: 训练数据的占比，默认0.75
# random_state：随机数种子，默认为None，使用当前的时间戳；非None值，可以保证多次运行的结果是一致的。
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8, random_state=1)
print(f'训练数据X的格式:{train_x.shape}')
print(f'测试数据X的格式:{test_x.shape}')

# 5、特征工程：正则化、标准化、文本的处理
# NOTE: 此处不做特征工程，后续有详细介绍

# 6、构建模型
knn_iris = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='kd_tree')
# 7、训练模型
knn_iris.fit(train_x, train_y)
# 8、模型效果的评估 （效果不好，返回第二步进行优化，达到要求）
pred_train = knn_iris.predict(train_x)
print(f'knn02模型训练集预测结果:{pred_train}')
acc_train = knn_iris.score(train_x, train_y)
print(f"knn02模型：训练集上的精度:{acc_train}")
pred_test = knn_iris.predict(test_x)
print(f'knn02模型预测结果:{pred_test}')
acc_test = knn_iris.score(test_x, test_y)
print(f"knn02模型：测试集上的效果(准确率):{acc_test}")

# 9、模型保存,方便后续调用
joblib.dump(knn_iris, "./knn_iris.m")  # 存储在当前文件夹