import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#导入数据
data=np.genfromtxt('E:\\机器学习\线性、非线性回归\Delivery.csv',delimiter=',')
x_data=data[:,:-1]
y_data=data[:,-1]
print(x_data)

model=linear_model.LinearRegression()   #线性模型
model.fit(x_data,y_data)                #训练完毕

#查看系数，截距
print('系数：',model.coef_)
print('截距：',model.intercept_)
#测试
x_test=[[102,4]]
predict=model.predict(x_test)
print('predict:',predict)

#画3D图
ax = plt.figure().add_subplot(111, projection='3d')  # 3d图的对象
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 形成网格矩阵
x0, x1 = np.meshgrid(x0, x1)  # 构成点（x0，x1）
z = model.intercept_ + model.coef_[0] * x0 + model.coef_[1] * x1  # (x0,x1,z)
# 画3d图
ax.plot_surface(x0, x1, z)
# 设置坐标
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()