import numpy
import pandas as pd
import random
import matplotlib.pyplot as plt
class Perceptron():
    def __init__(self):
        self.weights=[]
    
    def fit(self,x,y,learning_rate=0.01,iterations=100):
        rows,feat=x.shape
        
        self.weights=numpy.random.rand(feat+1)
        
        #TRAINING ALGORITHM
        for i in range(iterations):
            ri=random.randint(0,rows-1)
            row=x[ri,:]
            ypred=self.predict(row)
            error=y[ri]-ypred
            self.weights[0]+=learning_rate*error
            
            for fi in range(feat):
                self.weights[fi]+=learning_rate*error*row[fi]
            
            if feat ==2 and i==iterations-1:
                w1 = self.weights[1]
                w2=self.weights[2]
                b=self.weights[0]
                x2=numpy.linspace(-25,25,50)
                y2=[-1*((w1*i)+b)/w2 for i in x2]
                plt.plot(x2,y2)
                plt.ylim(-25,25)
                # plt.show()
            
            total_error=0
            for ri in range(rows):
                row=x[ri,:]
                ypred=self.predict(row)
                error=y[ri]-ypred
                total_error+=error**2
            mse=total_error/rows
            print('Iteration',i,'with error',mse)
            
    def predict(self,row):
        act=self.weights[0]
        for weight,feature in zip(self.weights[1:],row):
            act+=weight*feature
            
        if act>=0:
            return 1.0
        return 0.0
    
random.seed(50)
data=[[],[],[],[]]
for i in range(4):
    if i==0:
        data[i]=[random.randint(0,25) for n in range(50)]
    elif i==2:
        data[i]=[random.randint(-25,0) for n in range(50)]
    else:
        data[i]=[random.randint(-25,25) for n in range(50)]

x=[]

for i in range(50):
    x.append([data[0][i],data[1][i],1])
    x.append([data[2][i],data[3][i],0])
    
random.shuffle(x)
x=numpy.array(x)
y=x[:,2]
[x1,y1,x2,y2]=[numpy.array(p) for p in data]
plt.plot(x1,y1,'r.')
plt.plot(x2,y2,'b.')
# plt.show()


pcp=Perceptron()
pcp.fit(x[:,0:2],y,learning_rate=0.1,iterations=10000)
plt.show()
