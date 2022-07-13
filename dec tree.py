import numpy as np
import pandas as pd

data = pd.read_csv('treedata.csv')
data.head()

#gini index
def gini_index(groups):
    count1 = len([entry for entry in groups if (entry[0] == 1 and entry[1] == 1)])
    count2 = len([entry for entry in groups if (entry[0] == 1 and entry[1] == 0)])
    count3 = len([entry for entry in groups if (entry[0] == 0 and entry[1] == 1)])
    count4 = len([entry for entry in groups if (entry[0] == 0 and entry[1] == 0)])
    total=count1+count2+count3+count4
    gini1=1-((count1/(count1+count2))**2)-((count2/(count1+count2))**2)
    gini2=1-((count3/(count3+count4))**2)-((count4/(count3+count4))**2)
    gini=((count1+count2)*gini1+(count3+count4)*gini2)/total
    return gini



def max(a,b,c):
    l=[a,b,c]
    max=0
    for i in range(3):
        if l[i]>l[max]:
            max=i
    return l[max]
#Blocked Artery gini index calculation
blocked_artery_grp=[[data.iloc[c,0],data.iloc[c,3]] for c in range(1000)]
g1=gini_index(blocked_artery_grp)
print('\nGini index for blocked artery patients=',g1)
print()


#Chest pain gini index calculation
chest_pain_grp=[[data.iloc[c,1],data.iloc[c,3]] for c in range(1000)]
g2=gini_index(chest_pain_grp)
print('\nGini index calulation for chest pain patients=',g2)
print()


#Good blood circulation gini index calculation
blood_circ_grp=[[data.iloc[c,2],data.iloc[c,3]] for c in range(1000)]
g3=gini_index(blood_circ_grp)
print('\nGini index for good blood circulation patients=',g3)
