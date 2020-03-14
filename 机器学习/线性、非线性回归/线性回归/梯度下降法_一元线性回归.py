#一种属性x叫一元
import numpy as np
import matplotlib.pyplot as plt

#导入数据
data=np.genfromtxt('E:\机器学习\线性、非线性回归\data.csv',delimiter=',')
x_data=data[:,0]        #第0列
y_data=data[:,1]        #第1列
plt.scatter(x_data,y_data)  #绘制散点
plt.show()

#学习率
lr=0.0001
#截距
b=0
#斜率
k=0
#最大迭代次数
epochs=50

#最小二乘法,代价函数(代价函数值最小时，回归程度最高)
def computer_error(b,k,x_data,y_data):
    totalError=0
    for i in range(0,len(x_data)):
        totalError+=(y_data[i]-(k * x_data[i] + b))**2
    return totalError/float(len(x_data))                #返回了真实值和预测值的误差平方和

#梯度下降算法，针对二元函数
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    #m是数据量的个数
    m=float(len(x_data))
    #循环epchos次
    for i in range(epochs):                     #循环训练epochs次 k和 b
        b_grad=0    #梯度下降的截距值
        k_grad=0    #梯度下降的斜率值
        #计算斜率和截距的下降梯度总和后求平均
        for j in range(0,len(x_data)):          #每次训练后的 k和 b形成的模型对每个点求梯度，求出新的 k和 b
            b_grad += (1/m) * ( (k * x_data[j] +b) - y_data[j])         #每个点的梯度和求平均
            k_grad += (1/m) * x_data[j] * ( (k * x_data[j] +b ) - y_data[j])
        #更新b，k
        b-=lr*b_grad
        k-=lr*k_grad
        #每迭代5次，输出一次图像
        # if i%5 == 0:
        #     print('epochs:',i)
        #     plt.plot(x_data,y_data,'b.')
        #     plt.plot(x_data,k*x_data+b,'r')
        #     plt.show()
    return b,k

if __name__=='__main__':
    print('开始b={}，k={}，error={}'.format(b,k,computer_error(b,k,x_data,y_data)))
    print('Running。。。')
    b,k=gradient_descent_runner(x_data,y_data,b,k,lr,epochs)
    print('训练完{}次后，b={}，k={}，error={}'.format(epochs,b,k,computer_error(b,k,x_data,y_data)))
    #绘图
    plt.plot(x_data,y_data,'b.')  #plot画图，传入x和y形成的点，b是blue，‘.’代表化成点
    plt.plot(x_data,k*x_data+b,'r') #绘制曲线
    plt.show()