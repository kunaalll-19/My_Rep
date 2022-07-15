import pandas as pd
from sklearn.metrics import accuracy_score


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


def tree(row):
    if row['Blocked Arteries'] == 1 :
        if row['Chest_Pain'] == 1 :
            row['Heart Disease Pred']=1
        else:
            row['Heart Disease Pred']=0
    else:
        if row['Good Blood Circulation']==0:
            row['Heart Disease Pred']=1
        else:
            row['Heart Disease Pred']=0
    return row

#################################################################################################################


#FOR CALCULATING GINI INDEX
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


#################################################################################################################


#MAKING SPLITS BASED ON BLOCKED OR UNBLOCKED ARTERIES
not_blocked=data[data['Blocked Arteries'] == 0]
blocked=data[data['Blocked Arteries'] == 1]


#PATIENTS WITH CHEST PAIN AND UNBLOCKED ARTERIES
chest_pain_not_blocked_grp=[[not_blocked.iloc[c,1],not_blocked.iloc[c,3]] for c in range(len(not_blocked))]
print('Gini index for patients with unblocked arteries and chest pain=',gini_index(chest_pain_not_blocked_grp))


#PATIENTS WITH GOOD BLOOD CIRCULATION AND UNBLOCKED ARTERIES
bld_circ_not_blocked_grp=[[not_blocked.iloc[c,2],not_blocked.iloc[c,3]] for c in range(len(not_blocked))]
print('Gini index for patients with good blood circulation and unblocked arteries=',gini_index(bld_circ_not_blocked_grp))


#PATIENTS WITH CHEST PAIN AND BLOCKED ARTERIES
chest_pain_blocked_grp=[[blocked.iloc[c,1],blocked.iloc[c,3]] for c in range(len(blocked))]
print('Gini index for patients with unblocked arteries and chest pain=',gini_index(chest_pain_blocked_grp))


#PATIENTS WITH GOOD BLOOD CIRCULATION AND BLOCKED ARTERIES
bld_circ_blocked_grp=[[blocked.iloc[c,2],blocked.iloc[c,3]] for c in range(len(blocked))]
print('Gini index for patients with good blood circulation and unblocked arteries=',gini_index(bld_circ_blocked_grp))


#################################################################################################################

#FINDING ACCURACY
test_data = pd.read_csv('treetest.csv')
test_data.head()

test_data = test_data.apply(tree, axis = 1)
test_data.head()

print(accuracy_score(test_data['Heart Disease'], test_data['Heart Disease Pred']))

#################################################################################################################