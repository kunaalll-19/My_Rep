import torch
import torch.nn as nn
import numpy
from sklearn import datasets
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
x,y=iris.data,iris.target

sample,features=x.shape
print(sample,features)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1234)

#SCALE
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

x_train=torch.from_numpy(x_train.astype(numpy.float32))
x_test=torch.from_numpy(x_test.astype(numpy.float32))
y_train=torch.from_numpy(y_train.astype(numpy.float32))
y_test=torch.from_numpy(y_test.astype(numpy.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)


#MODEL
class LR(nn.Module):
    def __init__(self,n_input_features):
        super(LR,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
    
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred

model=LR(features)

#LOSS AND OPTIMISATION
learning_rate=0.01
criterion=nn.BCELoss()
optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)

#TRAINING LOOP
num=100
for i in range(num):
    #forward pass
    y_predicted=model(x_train)
    loss=criterion(y_predicted,y_train)

    #backward pass
    loss.backward()
    
    #update weights 
    optimiser.step()
    optimiser.zero_grad()
    
    #plot
    if (i+1)%10==0:
        print('epoch:',i+1,'loss:',loss.item())
        
        
#ACCURACY CHECK
with torch.no_grad():
    y_predicted=model(x_test)
    y_pred_cls=y_predicted.round()
    acc=y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print('accuracy=',acc)