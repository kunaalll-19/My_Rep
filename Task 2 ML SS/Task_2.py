import pandas as pd
import numpy
import matplotlib.pyplot as plt

data=pd.read_csv('C:/Users/Kunal Bibolia/Summer-School-2022/Summer-School-2022/Task_2/data/train_data.csv')
col=list(data.columns)

class model():
    def __init__(self):
        self.price_list=list(data['price'])
        self.ab_list=list(data['abtest'])
        self.vtype_list=list(data['vehicle_type'])
        self.yor_list=list(data['year_of_registration'])
        self.gear_list=list(data['gearbox'])
        self.power_list=list(data['power'])
        self.model_list=list(data['model'])     
        self.fuel_list=list(data['fuel_type'])     
        self.km_list=list(data['kilometer'])     
        self.mor_list=list(data['month_of_registration'])     
        self.brand_list=list(data['brand'])     
        self.nrd_list=list(data['not_repaired_damage'])     
        # self.postal_list=list(data['postal_code'])
        
    def assign_weights(self):
        total=0
        for i in self.price_list:
            total+=i
            weights=[]
            for i in range(len(self.price_list)):
                weigh=(self.price_list[i]/total)
                weights.append(weigh)
            weights=numpy.array(weights)
            return weights

