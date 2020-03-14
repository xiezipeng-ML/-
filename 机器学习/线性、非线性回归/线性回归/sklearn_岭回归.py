# Longley数据集，来自J.W.Longly（1967）发表在JASA上的一篇论文，是强线性的宏观经济数据，包含GNP deflator(GNP平减指数)、
# GNP(国民生产总值）、Unemployed（失业率）、ArmedForce（武装力量）、Population、year、employed，因为它存在严重的共线性问题，
# 所以。。。。
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#读数据
data=np.genfromtxt('E:\机器学习\线性、非线性回归\longley.csv',delimiter=',')
#print(data)

x_data=data[1:,2:]
y_data=data[1:,1]
# print(x_data)
# print(y_data)

#创建模型
alphas_to_test=np.linspace(0.001,1)         #设置50个岭回归损失函数的惩罚项的系数
# Ridge是岭回归，CV（cross validation）是交叉验证法，alphas是惩罚项系数，store_cv_values是否存储验证的一些结果
#用交叉验证法验证惩罚项的回归系数
model=linear_model.RidgeCV(alphas=alphas_to_test,store_cv_values=True)
model.fit(x_data,y_data)

#输出训练后得到的最好的岭系数
print(model.alpha_)
#训练过程中的loss值 得到一个16*50的矩阵，16是交叉验证法的16次验证，50是设置的50个惩罚项系数的每个系数得到的loss值
print(model.cv_values_.shape)

#画图
#岭系数（惩罚项参数）和loss值的关系
plt.plot(alphas_to_test,model.cv_values_.mean(axis=0))      #.mean是求平均值，axis是求平均值时候的方向，等于0是求16的平均
                                                            #每个惩罚项系数得到16个loss值
plt.plot(model.alpha_,min(model.cv_values_.mean(axis=0)),'ro')  #model.alpha_是得到最好的岭系数(横轴)
                                                                #纵轴是loss值中最小的
plt.show()

print( model.predict(x_data[2,np.newaxis]) )            #使用第二行数据验证
