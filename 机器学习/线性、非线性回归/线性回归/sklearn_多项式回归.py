#n元多项式有n个属性多个项
#多项式： y= a0+ a1xi + a2xi^2 +.....anxi^k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression           #线性模型包
from sklearn.preprocessing import PolynomialFeatures        #多项式包

#载入数据
data=np.genfromtxt('E:\\机器学习\线性、非线性回归\job.csv',delimiter=',')
x_data=data[1:,1]
y_data=data[1:,2]
plt.scatter(x_data,y_data)
plt.show()
print(x_data)                       #这里的x_data是个列表

x_data=data[1:,1,np.newaxis]        #fit接口要求数据是矩阵形式
y_data=data[1:,2,np.newaxis]
model=LinearRegression()
model.fit(x_data,y_data)
print(x_data)

plt.plot(x_data,y_data,'b.')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()

#由于要求多项式分布，而x_data数据是,定义多项式回归，degree的值可以调节多项式的特征
#degree=1 : y=a1*x^0 + a2*x^1             一元一次方程
#degree=2 : y=a1*x^0 + a2*x^1 + a3*x^2    一元二次方程
#。。。 自动训练 ai
poly_reg=PolynomialFeatures(degree=3)  #多项式对象       degree越大，模型越复杂，考虑越周全
#特征处理，改变数据特征
x_poly=poly_reg.fit_transform(x_data)
print(x_poly)
#定义线性回归模型
lin_reg=LinearRegression()
#训练模型
lin_reg.fit(x_poly,y_data)

#画图
plt.plot(x_data,y_data,'b.')
plt.plot(x_data,lin_reg.predict(poly_reg.fit_transform(x_data)),c='r')
plt.title('Wage CHanges')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#取更多的x值让曲线平滑
plt.plot(x_data,y_data,'b.')
x_test=np.linspace(1,10,100)  #生成1到10内100个均匀分布的点
print(x_test)           #数列
x_test=x_test[:,np.newaxis]
plt.plot(x_test,lin_reg.predict(poly_reg.fit_transform(x_test)),c='r')
plt.title('Wage CHanges')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()