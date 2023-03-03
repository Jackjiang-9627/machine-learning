import numpy as np
import pandas as pd
# 训练集数据 [亲吻镜头次数，打斗镜头次数，电影类型（1：爱情，-1：动作）]
T = [[3, 104, -1], [2, 100, -1], [1, 81, -1],
     [101, 10, 1], [99, 5, 1], [98, 2, 1]]
# 预测数据，判断其电影类型
x_test = [16, 94]
# 邻近点
k = 5
distance = [(np.sum((np.array(i[:-1])-np.array(x_test)) ** 2) ** 0.5, i[-1]) for i in T]
distance.sort()
print(distance)
dis_df = pd.DataFrame(distance[:k], columns=['distance', 'labels'])
weights_sum = np.sum(pd.Series(1 / dis_df['distance']))

dis_df['weight'] = (1 / dis_df['distance']) / weights_sum
print(dis_df)
if np.sum(dis_df[dis_df['labels'] == 1]['weight']) > np.sum(dis_df[dis_df['labels'] == -1]['weight']):
    pred = '爱情'
else:
    pred = '动作'
print(f'(16,94)预测为{pred}片')