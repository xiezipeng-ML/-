import numpy as np
from sklearn import linear_model

data=np.genfromtxt('E:\\机器学习\线性回归以及非线性回归\longley.csv',delimiter=',')
x_data=data[1:,2:]
y_data=data[1:,1]

print(data)

model=linear_model.LassoCV()
model.fit(x_data,y_data)

print(model.alpha_)     #LASSO参数
print(model.coef_)      #特征系数

print(model.predict(x_data[-2,np.newaxis]))