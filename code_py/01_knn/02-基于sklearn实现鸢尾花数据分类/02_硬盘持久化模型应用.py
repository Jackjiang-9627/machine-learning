import joblib
import warnings
warnings.filterwarnings("ignore")
# 1、加载模型,注意模型文件存储路径
knn = joblib.load('./knn_iris.m')
# 2、预测数据
x = [[3.3, 2.1, 4.7, 1.9]]
y_hat = knn.predict(x)
y_hat_prob = knn.predict_proba(x)
print(f'预测为：{y_hat[0]}')
