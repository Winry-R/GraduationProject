# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:54:50 2019

@author: Winry
"""

import os
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
filePath = 'C:\\Users\\Winry\\.spyder-py3\\毕业设计\\特征选择\\On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2017_5'

dataCsv = os.listdir(filePath)

adict = locals()
lst1 = [1,2,3,4,5,6,7,8,9,10,11,12]
for i,s in enumerate(lst1):
    adict['df%s' % (i+1)] = pd.read_csv(dataCsv[i])


for i in range(len(lst1)):
    adict['df%s' % (i+1)] = adict['df%s' % (i+1)].iloc[:,0:56]
    
    

deleteByte = ['Year','FlightDate','Reporting_Airline','IATA_CODE_Reporting_Airline',
 'Tail_Number','OriginAirportID','OriginCityName','OriginState','OriginStateName',
 'DestAirportID','Dest','DestCityName','DestState','DestStateName','DepDelayMinutes',
 'DepDel15','DepTimeBlk','ArrDelayMinutes','ArrTimeBlk','CancellationCode',
 'ArrivalDelayGroups','ArrDel15','CRSArrTime']

specificAirport = ['ORD','ATL','JFK','LAX','SFO','SEA']
for i in range(len(lst1)):
    temp = []
    
    adict['df17_%s' % (i+1)] = adict['df%s' % (i+1)].drop(deleteByte,axis = 1)
    
    interVariables = adict['df17_%s' % (i+1)]['Origin']
    
    for j in range(len(interVariables)):
        if interVariables.iloc[j] in specificAirport:
            temp.append(j)
    adict['index_%s' % (i+1)] = temp
    adict['df17_%s' % (i+1)] = adict['df17_%s' % (i+1)].iloc[temp,:]
    adict['df17_%s' % (i+1)] = adict['df17_%s' % (i+1)].dropna()
    adict['Delay17_%s' % (i+1)] = adict['df17_%s' % (i+1)]['ArrDelay']
    adict['df17_%s' % (i+1)] = adict['df17_%s' % (i+1)].drop('ArrDelay',axis = 1)

df2017 = df17_1
Delay2017 = Delay17_1
for i in range(len(lst1)-1):
    df2017 = pd.concat([df2017,adict['df17_%s' % (i+2)]])
    Delay2017 = pd.concat([Delay2017,adict['Delay17__%s' % (i+2)]])
    
    
data2017 = pd.concat([df2017,Delay2017],axis=1)
'''
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)

n是要抽取的行数。（例如n=20000时，抽取其中的2W行）

frac是抽取的比列。（有一些时候，我们并对具体抽取的行数不关系，我们想抽取其中的百分比，这个时候就可以选择使用frac，例如frac=0.8，就是抽取其中80%）

replace：是否为有放回抽样，取replace=True时为有放回抽样。

'''

tarinData2017 = data2017.sample(n=20000)

x_train, x_test, y_train, y_test = train_test_split(
tarinData2017.iloc[:,0:32], tarinData2017.iloc[:,-1], test_size=0.5, random_state=0)

