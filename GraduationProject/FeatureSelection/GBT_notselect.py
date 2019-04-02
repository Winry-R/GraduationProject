# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:19:06 2019

@author: Winry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GMM
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")

f = open('data0118.csv', 'rb')
df = pd.read_csv(f)
f.close()

data = df.values
X = data[:,0:-1]
y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = GradientBoostingClassifier(random_state=10)
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)
accuracyTemp = clf.score(x_test, y_test)

mod_logistic = logisticRe()
mod_logistic.fit(x_train,y_train)
mod_logistic.gradPlot()
y_pre = mod_logistic.predict(x_test)
Score = mod_logistic.value_Score(y_test)
for i ,x in enumerate(y_train):
    print(i)


f = open('data_new118.csv', 'rb')
d1 = pd.read_csv(f)
f.close()
d1 = d1.values
x_new = d1[:,0:-1]
y_new = d1[:,-1]
accuracyTemp = clf.score(x_new, y_new)
