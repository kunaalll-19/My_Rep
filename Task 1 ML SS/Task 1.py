#TASK 1 ML/DL SUMMER SCHOOL
import matplotlib.pyplot as mp
import pandas as pd
import csv

def max(l):
    max=l[0]
    for i in range(len(l)):
        if max<l[i] :
            max=l[i]
    return max

def min(l):
    min=l[0]
    for i in range(len(l)):
        if min<l[i] :
            min=l[i]
    return min

def str_to_float(string):
    str1=''
    for i in string :
        if ((not(i.isspace())) and (i!='%')):
            str1+=i
    if '%' in str1 :
        str1.replace('%','')
        print(str1)
    return float(str1)
    

dat = pd.read_csv('data.csv')


'''Required variables'''
columns = []   #Columns
rows = []      #Row values
eid = []       #Employee IDs
tenure = []    #Tenures of Employees
age = []       #Age of Employees
rating = []    #Last ratings of the Employees
incomes = []   #Income of Employees
eng = []       #Engagement score of Employees
male = 0       #Male Employees
female = 0     #Female Employees
bach = 0       #Bachelor Employees
mba = 0        #MBA Employees
div = 0        #Divorced Employees
single = 0     #Single Employees
marr = 0       #Married Employees
north = 0      #Employees in Northern Zone
east = 0       #Employees in Eastern Zone
west = 0       #Employees in Western Zone
south = 0      #Employees in Southern Zone
center = 0     #Employees in Central Zone



with open('data.csv','r+') as data:
    cread=csv.reader(data)
    columns=next(cread)
    for row in cread :
        rows.append(row)

for i in range(len(columns)):
    for j in rows:
        if columns[i]=='EmpID' :
            eid.append(j[i])
        elif columns[i]=='Tenure' :
            tenure.append(str_to_float(j[i]))
        elif columns[i]=='Gender' :
            if ('Male'in j[i]) :
                male+=1
            else:
                female+=1
        elif columns[i]=='Education' :
            if ('Bachelors'in j[i]) :
                bach+=1
            else:
                mba+=1
        elif columns[i]=='Age' :
            age.append(str_to_float(j[i]))
        elif columns[i]=='Last Rating' :
            rating.append(str_to_float(j[i]))
        elif columns[i]=='Monthly Income' :
            incomes.append(str_to_float(j[i]))
        elif columns[i]=='Engagement Score (% Satisfaction)' :
            eng.append(str_to_float(j[i]))
        elif columns[i]=='Marital Status' :
            if ('Divorced'in j[i]) :
                div+=1
            elif ('Single'in j[i]) :
                single+=1
            else:
                marr+=1
        elif columns[i]=='Zone' :
            if ('North'in j[i]) :
                north+=1
            elif ('East'in j[i]) :
                east+=1
            elif ('West'in j[i]) :
                west+=1
            elif ('South'in j[i]) :
                south+=1
            else:
                center+=1

'''HISTOGRAMS'''
#Displaying tenure variation
mp.xlabel('Tenures')
mp.ylabel('Number of Employees')
mp.title('Tenure Variations')
mp.hist(tenure,color='r',edgecolor='w')
mp.show()


#Displaing age variations
mp.xlabel('Age')
mp.ylabel('Number of Employees')
mp.title('Age Variations')
mp.hist(age,color='b',edgecolor='w')
mp.show()


#Displaying rating variation
mp.xlabel('Last Rating')
mp.ylabel('Number of Employees')
mp.title('Rating Variations')
mp.hist(rating,color='g',edgecolor='w')
mp.show()


#Displaying income variation
mp.xlabel('Monthly Incomes')
mp.ylabel('Number of Employees')
mp.title('Inclome Variations')
mp.hist(incomes,color='y',edgecolor='w')
mp.show()


#Displaying engagement variations
mp.xlabel('Engagement Score (% Satisfaction)')
mp.ylabel('Number of Employees')
mp.title('Engagement Score Variations')
mp.hist(eng,color='c',edgecolor='w')
mp.show()


'''PIE CHARTS'''
#Gender variations
gendervar=[male,female]
mp.title('Gender Varitation')
mp.pie(gendervar,labels=['Male','Female'],autopct='%1.2f%%',explode=[0,0.5])
mp.show()


#Educational Variation
edu=[bach,mba]
mp.title('Educational Variation')
mp.pie(edu,labels=['Bachelors','MBA'],colors=['r','c'],autopct='%1.2f%%',explode=[0,0.5])
mp.show()


#Marital Status Variation
mar=[div,single,marr]
mp.title('Marital Status Variation')
mp.pie(mar,labels=['Divorced','Single','Married'],colors=['silver','green','blue'],autopct='%1.2f%%',explode=[0.1,0.1,0.1])
mp.show()


#Zonal Variation
zones=[north,east,west,south,center]
mp.title('Zonal Variation')
mp.pie(zones,labels=['North','East','West','South','Center'],autopct='%1.2f%%',explode=[0.2,0.2,0.2,0.2,0.2])
mp.show()