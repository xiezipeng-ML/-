import numpy as np
from sklearn import linear_model

data=np.genfromtxt('E:\\机器学习\线性回归以及非线性回归\longley.csv',delimiter=',')
x_data=data[1:,2:]
y_data=data[1:,1]
print(y_data)

modle=linear_model.ElasticNetCV()
modle.fit(x_data,y_data)

print(modle.alpha_)
print(modle.coef_)

print(modle.predict(x_data[-2,np.newaxis]))