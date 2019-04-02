# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:00:01 2019

@author: Winry
"""

import os
import pandas as pd

filePath = 'C:\\Users\\Winry\\.spyder-py3\\GraduationProject\\FeatureSelection\\2018\\'

dataCsv = os.listdir(filePath)


adict = locals()
lst1 = [1,2,3,4,5,6,7,8,9,10,11,12]
for i,s in enumerate(lst1):
    adict['df%s' % (i+1)] = pd.read_csv(filePath+dataCsv[i])


for i in range(len(lst1)):
    adict['df%s' % (i+1)] = adict['df%s' % (i+1)].iloc[:,0:56]
    
    

deleteByte = ['Year','FlightDate','Reporting_Airline','IATA_CODE_Reporting_Airline',
 'Tail_Number','OriginAirportID','OriginCityName','OriginState','OriginStateName',
 'DestAirportID','Dest','DestCityName','DestState','DestStateName','DepDelayMinutes',
 'DepDel15','DepTimeBlk','ArrDelayMinutes','ArrTimeBlk','CancellationCode',
 'ArrivalDelayGroups','ArrDel15','CRSArrTime','ArrTime']

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
    Delay2017 = pd.concat([Delay2017,adict['Delay17_%s' % (i+2)]])
    
    
data2017 = pd.concat([df2017,Delay2017],axis=1)
data2017 = data2017.drop(['Origin'],axis = 1)

data2017.to_csv('C:\\Users\\Winry\\.spyder-py3\\GraduationProject\\FeatureSelection\\data2018.csv',index=False)