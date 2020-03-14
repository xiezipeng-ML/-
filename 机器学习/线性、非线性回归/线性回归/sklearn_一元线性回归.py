# 对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、
# 分类(Classfication)、聚类(Clustering)等方法。
# Sklearn自带部分数据集，也可以通过相应方法进行构造

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#载入数据
data=np.genfromtxt('..\data.csv',delimiter=',')
x_data=data[:,0]
print(x_data)
y_data=data[:,1]
plt.scatter(x_data,y_data)
plt.show()
# print(x_data.shape)                 #x_data是一维

x_data=data[:,0,np.newaxis]         #对x_data进行维度扩展，形成二维的，因为 LinearRegression要求的数据格式
print(x_data)
y_data=data[:,1,np.newaxis]
#创建并且拟合模型
model=LinearRegression()            #线性模型对象
model.fit(x_data,y_data)            #将数据拟合上（建模）

plt.plot(x_data,y_data,'b.')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()