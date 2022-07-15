import pandas as pd

data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Session_4/data/temps.csv')
print(data.describe())
data=pd.get_dummies(data)
print(data.iloc[:,5:].head(5))