#两种属性x1，x2是两元
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#读取数据
data=np.genfromtxt('E:\\机器学习\线性、非线性回归\Delivery.csv',delimiter=',')
# print(data)
#切分数据
x_data=data[:,:-1]
y_data=data[:,-1]
print(x_data)
print(y_data)

#学习率
lr=0.0001
#参数
theta0=0
theta1=0
theta2=0
#迭代次数
epochs=1000

#最小二乘法
def computer_error(theta0,theta1,theta2,x_data,y_data):
    totalError=0
    for i in range(0,len(x_data)):
        totalError+=( y_data[i] - ( theta1 * x_data[i,0] + theta2 * x_data[i,1] + theta0))**2
    return totalError/float(len(x_data))

#梯度下降
def gradient_decent_runner(x_data,y_data,theta0,theta1,theta2,lr,epochs):
    m=float(len(x_data))
    for i in range(0,epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0,len(x_data)):
            theta0_grad+=(1/m)*( (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j] )
            theta1_grad+=(1/m)*( (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j] )*x_data[j,0]
            theta2_grad+=(1/m)*( (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j] )*x_data[j,1]
        theta0-=lr*theta0_grad
        theta1-=lr*theta1_grad
        theta2-=lr*theta2_grad
    return theta0,theta1,theta2

if __name__=='__main__':
    print('开始：theta0={},theta1={},theta2={},error={}'
          .format(theta0,theta1,theta2,computer_error(theta0,theta1,theta2,x_data,y_data)))
    print('Running...')
    theta0,theta1,theta2=gradient_decent_runner(x_data,y_data,theta0,theta1,theta2,lr,epochs)
    print('训练完{}次后：theta0={},theta1={},theta2={},error={}'
          .format(epochs,theta0,theta1,theta2,computer_error(theta0,theta1,theta2,x_data,y_data)))
    #画图
    ax=plt.figure().add_subplot(111,projection='3d')        #3d图的对象
    ax.scatter(x_data[:,0],x_data[:,1],y_data,c='r',marker='o',s=100)
    x0=x_data[:,0]
    x1=x_data[:,1]
    #形成网格矩阵
    x0,x1=np.meshgrid(x0,x1)    #构成点（x0，x1）
    z=theta0+theta1*x0+theta2*x1    #(x0,x1,z)
    #画3d图
    ax.plot_surface(x0,x1,z)
    #设置坐标
    ax.set_xlabel('Miles')
    ax.set_ylabel('Num of Deliveries')
    ax.set_zlabel('Time')
    plt.show()