import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('E:\机器学习\线性、非线性回归\longley.csv',delimiter=',')
#print(data)

x_data=data[1:,2:]
y_data=data[1:,1,np.newaxis]
#print(x_data)
#print(y_data)

print(np.mat(x_data).shape)
print(np.mat(y_data).shape)

#给样本添加偏置项
add_matrix=np.ones((16,1))
X_data=np.concatenate((add_matrix,x_data),axis=1)
print(X_data.shape)

#岭回归标准方法
def weight(xArr,yArr,lam=0.2):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xTx= xMat.T * xMat
    rxTx=xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(rxTx) == 0.0:
        print('This matrix cannot do inverse')
    ws=rxTx.I*xMat.T*yMat                               #求得的特征系数矩阵
    return ws

ws=weight(X_data,y_data)
print(ws)
print('---------')
#计算预测值
print(np.mat(X_data)*np.mat(ws))