import pandas as pd
from sklearn import neighbors
# 训练集数据
T = [[3, 104, -1], [98, 2, 1], [2, 100, -1],
     [101, 10, 1], [99, 5, 1], [1, 81, -1]]
data = pd.DataFrame(T, columns=['亲吻镜头次数', '打斗镜头次数', '电影类型'])
train_x, train_y = (data.iloc[:, :-1], data.iloc[:, -1])
# 预测数据，判断其电影类型
test_x = pd.DataFrame([[18, 90], [50, 50]],
                      columns=['亲吻镜头次数', '打斗镜头次数'])
# 邻近点数
k = 3
# 实例化k近邻分类对象
knn01 = neighbors.KNeighborsClassifier(n_neighbors=k)
# 训练模型，存储训练集数据
knn01.fit(train_x, train_y)
# 预测
pred = knn01.predict(test_x)
print(f'预测结果为:{pred}')
# 精度accuracy
acc = knn01.score(X=train_x, y=train_y)
print(f'预测精度为：{acc:.4f}')