import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report       #含有评估模型的工具(计算准确率，召回率，F1 )
from sklearn import preprocessing                       #数据标准化工具

scale=False                         #是否数据标准化

data=np.genfromtxt('E:\机器学习\线性、非线性回归\LR-testSet.csv',delimiter=',')
x_data=data[:,:-1]
y_data=data[:,-1]

def plot():
    x0=[]       #存放0类别的x特征
    x1=[]       #存放1类别的x特征
    y0=[]       #存放0类别的y特征
    y1=[]       #存放1类别的y特征

    #特征数据分类存放
    for i in range(len(x_data)):
        if y_data[i]==0:
            x0.append(x_data[i,0])
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])
    #画图
    scatter0=plt.scatter(x0,y0,c='b',marker='o')
    scatter1=plt.scatter(x1,y1,c='r',marker='x')
    #画图例（说明）
    plt.legend(handles=[scatter0,scatter1],labels=['kind:0','kind:1'],loc='best')

plot()
plt.show()

#要找出一条线分割两种类型  例如直线：a0x0 + a1x1 + b = 0
y_data=data[:,-1,np.newaxis]
print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
#添加模型的偏置项
X_data=np.concatenate((np.ones((100,1)),x_data),axis=1)
print(X_data.shape)

def sigmod(x):
    return 1.0/(1+np.exp(-x))

def cost(xMat,yMat,ws):
    left=np.multiply(yMat,np.log(sigmod(xMat*ws)))
    right=np.multiply(1-yMat,np.log(1-sigmod(xMat*ws)))
    return np.sum(left+right)/-(len(xMat))

def gradAscent(xArr,yArr):
    if scale==True:
        xArr=preprocessing.scale(xArr)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)

    lr=0.001
    epochs=10000
    costList=[]

    m,n=np.shape(xMat)
    ws=np.mat(np.ones((n,1)))               #ws系数矩阵从每个数从1开始训练

    for i in range(epochs+1):
        h=sigmod(xMat*ws)
        ws_gead=xMat.T*(h-yMat)/m
        ws=ws-lr*ws_gead

        if i%50==0:
            costList.append(cost(xMat,yMat,ws))
    return ws,costList

ws,costList=gradAscent(X_data,y_data)
print(ws)
print(costList)
if scale==False:
    plot()
    x_test=[[-4],[3]]
    #决策边界函数
    y_test=(-ws[0] - x_test*ws[1])/ws[2]
    plt.plot(x_test,y_test,'k')
    plt.show()

x=np.linspace(0,10000,201)
plt.plot(x,costList,c='r')
plt.title('Train')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

def predict(x_data,ws):
    if scale==True:
        x_data=preprocessing.scale(x_data)
    xMat=np.mat(X_data)
    return [1 if x>=0.5 else 0 for x in sigmod(xMat*ws)]

prediction=predict(X_data,ws)
print(classification_report(y_data,prediction))