import pandas as pd
import numpy
import matplotlib.pyplot as plt

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


from matplotlib import animation,rc
from display import HTML

fig,ax=plt.subplots()
plt.plot(x,y,'ro')
line, =ax.plot([],[],width=2)
def animate(i):
    x = numpy.arange(20,90)
    y = m_hist[i] * x + c_hist[i]
    line.set_data(x, y)
    return (line,)


anim = animation.FuncAnimation(fig, animate,frames=len(m_hist), interval=100)
HTML(anim.to_html5_video())