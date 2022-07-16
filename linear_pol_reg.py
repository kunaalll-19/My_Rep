import pandas as pd
import numpy
import matplotlib.pyplot as plt
'''
data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Session_2/data/data.csv')
data.head()

#PLOTTING THE VALUES
x=[data.iloc[i,0] for i in range(99)]
x=numpy.array(x)
y=[data.iloc[i,1] for i in range(99)]
y=numpy.array(y)
plt.plot(x,y,'ro')
plt.show()


#HYPERPARAMETERS
iterations=50
lr=0.0001

m=0.0
c=0.0

hist_cost=[]
m_hist=[]
c_hist=[]

for i in range(iterations):
    y_pred=m*x+c
    error=y_pred-y
    cost=0.5*numpy.mean(error**2)
    
    m_der=numpy.mean((y_pred-y)*x)
    c_der=numpy.mean(y_pred-y)
    m-=lr*m_der
    c-=lr*c_der
    
    hist_cost.append(cost)
    m_hist.append(m)
    c_hist.append(c)
    
#PLOTTING HISTORY COST
x_val=numpy.arange(0,len(hist_cost),1)
plt.plot(x_val,hist_cost,'r')
plt.show()

#PLOTTING LINEAR REGRESSION OF THE FUNCTION
plt.plot(x,y,'ro')
x_line=numpy.linspace(20,80,1000000)
y_line=m*x_line+c
plt.plot(x_line,y_line)
plt.show()


from matplotlib import animation
# from Ipython.display import HTML

fig,ax=plt.subplots()
plt.plot(x,y,'ro')
line, =ax.plot([],[],linewidth=2)
def animate(i):
    x = numpy.arange(20,90)
    y = m_hist[i] * x + c_hist[i]
    line.set_data(x, y)
    return (line,)


anim = animation.FuncAnimation(fig, animate,frames=len(m_hist), interval=100)
# HTML(anim.to_html5_video())
'''


'''LINEAR REGRESSION USING SCIKIT
from sklearn.linear_model import LinearRegression

x = numpy.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
y = numpy.array([4, 23, 10, 12, 22, 35])

LRmodel=LinearRegression().fit(x,y)
#FINDING SLOPE AND INTERCEPT
print('intercept=',LRmodel.intercept_)
c=LRmodel.intercept_
print('slope=',LRmodel.coef_)
m=LRmodel.coef_

#PREDICTING Y VALUES
y_pred=LRmodel.predict(x)
for i in range(6):
    print('x=',x[i])
    print('y prediction=',y_pred[i])
plt.plot(x,y,'ro')
plt.plot(x,y_pred)
x_line=numpy.linspace(0,60,100)
y_line=m*x_line+c
plt.plot(x_line,y_line,'orange')
plt.show()
'''

'''
#POLYNOMIAL REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

#Making a 2D random array
x=4*numpy.random.rand(100,1)-2
y=4+x+2*x**2-5*x**3

#TRANSFORMING X INTO 2D ARRAY WITH COEFFICIENTS OF X
poly_feat=PolynomialFeatures(degree=3,include_bias=False)
x_poly=poly_feat.fit_transform(x)

reg=LinearRegression()
reg.fit(x_poly,y)

a0 = reg.intercept_[0]
a1 = reg.coef_[0][0]
a2 = reg.coef_[0][1]
a3 = reg.coef_[0][2]

plt.plot(x,y,'ro')
x_line=numpy.linspace(-2,2,10000)
y_line=a0+a1*x_line+a2*x_line**2+a3*x_line**3
plt.plot(x_line,y_line)
plt.show()
'''


#TRADEOFF B/W BIAS AND VARIANCE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator

#GETTING THE DATA POINTS
numpy.random.seed(0)
x=2-3*numpy.random.normal(0,1,20)
y=x-2*x**2+0.5*x**3-numpy.random.normal(-3,3,20)
plt.scatter(x,y,s=10)
plt.show()

x=x[:,numpy.newaxis]
y=y[:,numpy.newaxis]

#LINEAR MODEL
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)

plt.plot(x,y,'bo')
plt.plot(x,y_pred,color='r')
plt.show()

#POLYNOMIAL MODEL 
poly_feat=PolynomialFeatures(degree=4)
x_poly=poly_feat.fit_transform(x)

model=LinearRegression()
model.fit(x_poly,y)
y_poly_pred=model.predict(x_poly)

plt.scatter(x,y,s=10)
#Sort the values of x before line plot
sorted_axis=operator.itemgetter(0)
sorted_zip=sorted(zip(x,y_poly_pred),key=sorted_axis)
x,y_poly_pred=zip(*sorted_zip)
plt.plot(x,y_poly_pred,color='r')
plt.show()