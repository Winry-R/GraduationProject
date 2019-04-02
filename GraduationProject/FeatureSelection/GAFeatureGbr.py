# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:03:59 2019

@author: Winry
"""

import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager


class GaForest(object):

    def __init__(self,alpha = 1, mutProbability = 0.01, crossProbability = 0.8, groupSize = 100, maxIteration = 1000,encodelength=10):
        '''

        :param alpha: weighte
        :param mutProbability: Mutation rate
        :param crossProbability: Cross rate
        :param groupSize: Population size
        :param maxIteration: Maximum number of iterations
        :param encodelength: Code length
        '''

        self.alpha = alpha
        self.mutProbability = mutProbability
        self.crossProbability = crossProbability
        self.groupSize = groupSize
        self.maxIteration = maxIteration
        self.encodelength = encodelength
        self.optimalSolutions = []
        self.optimalValues = []
        self.chromosomes = np.ones((self.groupSize, (self.encodelength)))
        for i in range(self.groupSize):
            self.chromosomes[i, :] = np.random.randint(0, 2, (self.encodelength))

    #
    # def  originalgroup(self):
    #
    #     self.chromosomes = np.zeros((self.groupSize, (self.encodelength)))
    #     for i in range(self.groupSize):
    #         self.chromosomes[i, :] = np.random.randint(0, 2, (self.encodelength))

    def fitness(self, index):
        '''

        :param index: Training label
        :return: Accuracy
        '''
        # clf = RandomForestClassifier(class_weight='balanced', random_state=1)
        clf = GradientBoostingClassifier(random_state=10)
        temp = self.x_train[:, index]
        clf.fit(temp, self.y_train)
        accuracyTemp = clf.score(self.x_test[:, index], self.y_test)
        return accuracyTemp

    def function(self, x_train, y_train, x_test, y_test):
        '''
        Obtain cumulative probability and objective function value
        '''

        self.func = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        chromosomes = self.chromosomes

        for i in range(self.groupSize):
            index = []
            for j in range(self.encodelength):

                if chromosomes[i][j] == 1:
                    index.append(j)
            accuracyTemp = self.fitness(index)
            numTemp = chromosomes.shape[1] / len(index)
            self.func.append((1 - self.alpha) * numTemp + self.alpha * accuracyTemp)
        probability = self.func / np.sum(self.func)
        self.cumulativeProbability = np.cumsum(probability)

    def selection(self):
        '''
        Select a population and generate a new population
        :return:
        '''

        chromosomeRow, chromosomeColumn = self.chromosomes.shape
        self.newgroup = np.ones((chromosomeRow, chromosomeColumn))
        turnProbability = np.random.rand(chromosomeRow)
        for i, pro in enumerate(turnProbability):
            logical = self.cumulativeProbability >= pro
            index = np.where(logical == 1)
            self.newgroup[i, :] = self.chromosomes[index[0][0], :]

    def cross(self):
        '''
        Population mating
        '''

        newgroupRow, newgroupColumn = self.newgroup.shape
        num = int(newgroupRow * self.crossProbability)
        if num % 2 != 0:
            num += 1

        self.crossGroup = np.zeros((newgroupRow, newgroupColumn))
        index = random.sample(range(newgroupRow), num)
        for i in range(newgroupRow):
            if i in index:
                pass
            else:
                self.crossGroup[i, :] = self.newgroup[i, :]
        while len(index) > 0:
            countOne = index.pop()
            countTwo = index.pop()
            tempOne = []
            tempTwo = []
            intersection = random.randint(0, newgroupColumn)

            tempOne.extend(self.newgroup[countOne][0:intersection])
            tempOne.extend(self.newgroup[countTwo][intersection:newgroupColumn])

            tempTwo.extend(self.newgroup[countTwo][0:intersection])
            tempTwo.extend(self.newgroup[countOne][intersection:newgroupColumn])
            self.crossGroup[countOne] = tempOne
            self.crossGroup[countTwo] = tempTwo

    def mut(self):
        '''
        Mutation to get new individuals
        :return:
        '''

        self.mutGroup = np.copy(self.crossGroup)

        groupRow, groupColumn = self.crossGroup.shape
        geneNum = int(groupRow * groupColumn * self.mutProbability)

        mutationGeneIndex = random.sample(range(0, groupRow * groupColumn), geneNum)

        for gene in mutationGeneIndex:

            chromoIndex = gene // groupColumn
            geneIndex = gene % groupColumn

            if self.mutGroup[chromoIndex, geneIndex] == 0:
                self.mutGroup[chromoIndex, geneIndex] == 1
            else:
                self.mutGroup[chromoIndex, geneIndex] == 0
        self.chromosomes = self.mutGroup

    def calculation(self,x_train,y_train,x_test,y_test):

        # self.chromosomes = self.originalgroup()

        for i in range(self.maxIteration):
            self.function(x_train,y_train,x_test,y_test)
            self.selection()
            self.cross()
            self.mut()
            self.function(x_train,y_train,x_test,y_test)
            self.optimalValues.append(np.max(list(self.func)))
            index = np.where(self.func == max(list(self.func)))
            self.optimalSolutions.append(self.chromosomes[index[0][0], :])
        optimalValue = np.max(self.optimalValues)
        optimalIndex = np.where(self.optimalValues == optimalValue)
        optimalSolution = self.optimalSolutions[optimalIndex[0][0]]
        return optimalSolution, optimalValue

    def fitnessdraw(self):
        my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\STKAITI.TTF")
        x = range(self.maxIteration)
        plt.plot(x, self.optimalValues, '-r')
        plt.xlabel('迭代次数', fontproperties=my_font)
        plt.ylabel('适应度值', fontproperties=my_font)
        plt.show()


if __name__ == '__main__':
    X, y = make_blobs(n_samples=300, n_features=10, centers=5, random_state=0, cluster_std=10.0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    GAF = GaForest(encodelength=X.shape[1])
    [optimalSolution,optimalValue] = GAF.calculation(x_train,y_train,x_test,y_test)

    print(optimalValue)
    GAF.fitnessdraw()
