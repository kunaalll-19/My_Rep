import pandas as pd
import numpy
import matplotlib.pyplot as plt
# DATA PREPROCESSORS
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# MODELS
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB


data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Session_3/data/Iris.csv')
data=data[:100]
data=data.drop(['SepalWidthCm','PetalWidthCm'],axis=1)

#CHECKING THE DATA
print(data.head(),'\n')
print(data.tail(),'\n')
print(data.info())


#EXTRACTING THE DATA
target=numpy.array(data['Species'].apply(lambda x:1 if x=='Iris-setosa' else -1))
sepal=numpy.array(data['SepalLengthCm'])
petal=numpy.array(data['PetalLengthCm'])
x=numpy.transpose([sepal,petal])
verification_data=numpy.transpose([sepal,petal,target])
print('Shape of x data',x.shape)

#PLOTTING THE DATA
plt.scatter(sepal[:50],petal[:50],marker='+',color='blue',label='setosa')
plt.scatter(sepal[50:],petal[50:],marker='o',color='orange',label='versicolor')
plt.legend(loc='best')

#IMPLEMENTING SVM
x,y=shuffle(x,target,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9)
print('Train  shape',x_train.shape)

from sklearn.metrics import accuracy_score


clf=SVC(kernel='linear',probability=True)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print('Accuracy =',accuracy_score(y_test,y_pred))
print('Coefficients of class variable=',clf.coef_)
print('Intercept',clf.intercept_)
w1,w2=clf.coef_[0][0],clf.coef_[0][1]
c=clf.intercept_
m=(-w1/w2)
print('y=mx+c format where m=',m,'and c=',(-c/w2))
x=x_train[:,0]
y=m*x-(c/w2)
plt.plot(x,y,'red')
plt.show()

'''
#IMPLEMENTING SVM FROM SCRATCH
class svm_model():
    
    def __init__(self,xdata=x_train,ydata=y_train,lr=1e-2,lamb=1e-2,epochs=10000):
        self.x=xdata
        self.y=ydata
        self.lr=lr
        self.lambda_parameter=lamb
        self.epochs=epochs
        
        self.weights=numpy.zeros(xdata.shape[1])
        self.bias=0
    
    def constraint(self,x,y,w,b):
        constraint=((numpy.dot(x,w)+b)*y) >=1
        return constraint
    
    def weight_update(self,dw,db):
        self.weights-=self.lr*dw
        self.bias-=self.lr*db
    
    def get_dw_db(self,x,y,w,constraint):
        if constraint.all():
            db=0
            dw=self.lambda_parameter*w
            return dw,db
        else:
            db=-y
            dw=self.lambda_parameter*w-numpy.dot(y,x)
            return dw,db
        
    def plot_graph(self,w,b,vals):
        val=vals[:,0]
        m1=-w[0]/w[1]
        c=-b/w[1]
        print(val.shape,'\n',m1.shape,'\n',c.shape)
        y1 = val*m1+c
        plt.scatter(sepal[:50],petal[:50],marker='+',color='green',label="setosa")
        plt.scatter(sepal[50:],petal[50:],marker='o',color='orange',label="veriscolor")
        plt.plot(vals[:,0],y1,color='red')
        plt.show()
    
    def find_hyperplane(self):
        epochs=self.epochs
        for e in range(1,epochs):
            for index,vector in enumerate(self.x):
                constraint=self.constraint(vector,self.y[index],self.weights,self.bias)
                dw,db=self.get_dw_db(constraint,vector,self.y[index],self.weights)
                self.weight_update(dw,db)
            
            if e<=10:
                print(-self.weights[0]/self.weights[1],-self.bias/self.weights[1])
                self.plot_graph(self.weights,self.bias,self.x)


train_svm=svm_model()
train_svm.find_hyperplane()'''

class SVM_model():

  # intializing the data
  def __init__(self,xdata=x_train,ydata=y_train,lr=1e-2,lamb=1e-2,epochs=10000):
    self.x = xdata
    self.y = ydata
    self.lr = lr
    self.lambda_parameter = lamb
    self.epochs = epochs

    self.weights = numpy.zeros(xdata.shape[1])
    self.bias = 0

  def constraint(self,x,y,w,b): # for getting the constraint 
    constraint = (numpy.dot(x,w) + b)*y >= 1
    return constraint

  def weight_update(self,dw,db): # for updaring thhe weights
    self.weights -= self.lr*dw
    self.bias -= self.lr*db

  def get_dw_db(self,constraint,x,y,w): # for getting the gradients
    if constraint:
      db = 0
      dw = self.lambda_parameter * w
      return dw,db
    else :
      db = -y
      dw = (self.lambda_parameter*w - numpy.dot(y,x))
      return dw,db

  def plot_graph(self,w,b,vals): # plottig the data to visualize the results
      y1 = vals[:,0]*(-w[0]/w[1])-(b/w[1])
      plt.scatter(sepal[:50],petal[:50],marker='+',color='green',label="setosa")
      plt.scatter(sepal[50:],petal[50:],marker='o',color='orange',label="veriscolor")
      plt.plot(vals[:,0],y1,color='red')
      plt.show()
  
  def find_hyperplane(self): # impleemnting all the above functions together to train SVM
      e = 1
      epochs = self.epochs

      while e <= epochs:
        for index,vector in enumerate(self.x):
          constraint = self.constraint(vector,self.y[index],self.weights,self.bias)
          dw,db = self.get_dw_db(constraint,vector,self.y[index],self.weights)
          self.weight_update(dw,db)

        if e<=10:
              print(-self.weights[0]/self.weights[1],-self.bias/self.weights[1])
              self.plot_graph(self.weights,self.bias,self.x)
        e += 1
        
Train_SVM = SVM_model()
Train_SVM.find_hyperplane()




#IMPLEMENTING GAUSSIAN NB
data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Session_3/data/Iris.csv')
print(data.head())
feat=data.iloc[:,:4].values
species=data['Species'].values
feat_x,species_y=shuffle(feat,species,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(feat_x,species_y,test_size=0.2)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('Accuracy=',accuracy_score(y_test,y_pred)*100)




#IMPLEMENTING KNN
data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Session_3/data/Iris.csv')
print(data.head())
feat=data.iloc[:,:4].values
species=data['Species'].values
feat_x,species_y=shuffle(feat,species,random_state=42)
x_train,x_test,y_train,y_test_KNN=train_test_split(feat_x,species_y,test_size=0.2)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


from sklearn.neighbors import KNeighborsClassifier

k_val=int((120**0.5)*2)
k_list=[i for i in range(3,k_val*3)]
accuracy_list=[]

for i in range(3,3*k_val):
    classifier_KNN=KNeighborsClassifier()
    classifier_KNN.fit(x_train,y_train)
    y_pred_KNN=classifier_KNN.predict(x_test)
    accuracy_list.append(accuracy_score(y_test_KNN,y_pred_KNN))
plt.plot(k_list,accuracy_list,label='y_test accuracy')
plt.legend(loc='best')
plt.show()
print(accuracy_list)