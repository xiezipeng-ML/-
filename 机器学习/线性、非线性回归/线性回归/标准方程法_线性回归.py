import numpy as np
import matplotlib.pyplot as plt

#载入数据
data=np.genfromtxt("E:\机器学习\线性、非线性回归\data.csv",delimiter=',')
x_data=data[:,0,np.newaxis]
y_data=data[:,1,np.newaxis]
plt.scatter(x_data,y_data)
plt.show()
print(np.mat(x_data).shape)     #mat是数据对应的矩阵
print(np.mat(y_data).shape)

#给特征数据添加偏置项
#原数据是（x1,x2,x3...）,要求参数（a0,a1,a2,a3...），a0是截距，所以要添加一列全为1的偏置项
X_data=np.concatenate( (np.ones((100,1)),x_data ),axis=1 )      #concatente是合并两矩阵，np.ones（）会生成一个n*m的值全为1的矩阵，axis是合并方向
print(X_data[:3])

#标准方程法求解回归参数,求w矩阵
def weights(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xTx=xMat.T * xMat
    #进行一个判断，如果值为0，那么该矩阵没有逆矩阵
    if np.linalg.det(xTx) ==0.0:     #求矩阵对应的行列式的值
        print('This matrix cannot do inverse')
        return
    ws=xTx.I * xMat.T * yMat
    return ws

ws=weights(X_data,y_data)
print(ws)
#画图
plt.plot(x_data,y_data,'b.')
plt.plot(x_data,ws[0] + x_data * ws[1],'r')
plt.show()

#可以随便取两点画出这条线
x_test=np.array( [ [20],[80] ] )
y_test=ws[0] + x_test * ws[1]
plt.plot(x_test,y_test,'r')
plt.show()
