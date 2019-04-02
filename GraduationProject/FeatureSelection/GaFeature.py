# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 09:38:25 2019

@author: Winry
"""
import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split




def originalgroup(groupSize,encodelength):

    chromosomes = np.zeros((groupSize, (encodelength)))
    for i in range(groupSize):
        chromosomes[i,:] = np.random.randint(0, 2, (encodelength))
    return chromosomes



def function(group,chromosomes,label,x_test,y_test,alpha):
    func = []
    groupTest = []
    for i in range(len(chromosomes)):
        
        temp = []
        for j in range(len(chromosomes[i])):
            if chromosomes[i][j] == 1:
                temp.append(j)
        model = []
        model = SVR()
        groupTest = group[:,temp]
        model.fit(groupTest,label)
        accuracyTemp = model.score(x_test[:,temp],y_test)
        numTemp = chromosomes.shape[1]/len(temp)
        func.append((1-alpha)*numTemp+alpha*accuracyTemp)
    probability = func / np.sum(func)
    cumulativeProbability = np.cumsum(probability)
    
    return func,cumulativeProbability


def selection(chromosomes,cumulativeProbability):
    
    chromosomeRow,chromosomeColumn = chromosomes.shape
    newgroup = np.ones((chromosomeRow,chromosomeColumn))
    turnProbability = np.random.rand(chromosomeRow)
    for i,pro in enumerate(turnProbability):
        logical = cumulativeProbability >= pro
        index = np.where(logical == 1)
        newgroup[i,:] = chromosomes[index[0][0],:]
    return newgroup


def cross(newgroup,crosspProbability):
    
    newgroupRow,newgroupColumn = newgroup.shape
    num = int(newgroupRow*crosspProbability)
    if num %2 != 0:
        num += 1
    
    crossGroup = np.zeros((newgroupRow,newgroupColumn))
    index = random.sample(range(newgroupRow),num)
    for i in range(newgroupRow):
        if i in index:
            pass
        else:
            crossGroup[i,:] = newgroup[i,:]
    while len(index)>0 :
        countOne = index.pop()
        countTwo = index.pop()
        tempOne = []
        tempTwo = []
        intersection = random.randint(0,newgroupColumn)
        
        tempOne.extend(newgroup[countOne][0:intersection])
        tempOne.extend(newgroup[countTwo][intersection:newgroupColumn])

        tempTwo.extend(newgroup[countTwo][0:intersection])
        tempTwo.extend(newgroup[countOne][intersection:newgroupColumn])
        crossGroup[countOne] = tempOne
        crossGroup[countTwo] = tempTwo
    return crossGroup


def mut(crossGroup,mutProbability):
    
    mutGroup = np.copy(crossGroup)
    
    groupRow,groupColumn = crossGroup.shape
    geneNum = int(groupRow*groupColumn*mutProbability)
    
    mutationGeneIndex = random.sample(range(0,groupRow * groupColumn), geneNum)
    
    for gene in mutationGeneIndex:
        
        chromoIndex = gene // groupColumn
        geneIndex = gene % groupColumn
        
        if mutGroup[chromoIndex,geneIndex] == 0:
            mutGroup[chromoIndex,geneIndex] == 1
        else:
            mutGroup[chromoIndex,geneIndex] == 0
    return mutGroup
maxIteration =  500
def main(maxIteration):
    optimalSolutions = []
    optimalValues = []
    X,y = make_blobs(n_samples=1000,n_features=10,centers=5,random_state=0,cluster_std=10.0)
    group,x_test,label,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    alpha = 0.95
    mutProbability = 0.01
    
    crosspProbability = 0.8
    groupSize = 100
    encodelength = group.shape[1]
    chromosomes = originalgroup(groupSize,encodelength)
    for i in range(maxIteration):
        
        [func,cumulativeProbability] = function(group,chromosomes,label,x_test,y_test,alpha)
        newgroup = selection(chromosomes,cumulativeProbability)
        crossGroup = cross(newgroup,crosspProbability)
        mutGroup = mut(crossGroup,mutProbability)
        group = mutGroup
        [func,cumulativeProbability] = function(group,chromosomes,label,x_test,y_test,alpha)
        
        optimalValues.append(np.max(list(func)))
        index = np.where(func == max(list(func)))
        optimalSolutions.append(group[index[0][0], :])
        
    optimalValue = np.max(optimalValues)
    
