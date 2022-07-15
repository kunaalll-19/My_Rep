import pandas as pd

data=pd.read_csv('IPLtest.csv')

print('MC CULLOH PITTS NEURON MODEL\n')
threshold = 3
def step(sum):
    if sum<threshold:
        return 0
    else:
        return 1

def mcp_neuron(j):
    sum=0
    for i in range(1,5):
        sum+=data.iloc[j,i]
    return step(sum)

error=0
for i in range(8):
    print('Predicted Output=',mcp_neuron(i))
    print('Actual output=',data.iloc[i,5])
    print()
    error+=((mcp_neuron(i)-data.iloc[i,5])**2)
mse=error/8
print('MSE=',mse)
print("################################################################")



print('PERCEPTRON MODEL\n')
weights=[3,2,1,1]
def step2(w_sum):
    if w_sum<threshold:
        return 0
    else:
        return 1

def perceptron(j):
    w_sum=0
    for i in range(1,5):
        w_sum+=(data.iloc[j,i]*weights[i-1])
    return step2(w_sum)

error=0
for i in range(8):
    print('Predicted Output=',perceptron(i))
    print('Actual output=',data.iloc[i,5])
    print()
    error+=((perceptron(i)-data.iloc[i,5])**2)
mse=error/8
print('MSE=',mse)